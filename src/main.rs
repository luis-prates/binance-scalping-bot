mod backtester;
mod binance;
mod config;
mod indicators;
mod ml_model;
mod strategy;

use anyhow::Result;
use binance::BinanceClient;
use config::Config;
use indicators::Signal;
use log::{error, info, warn};
use rust_decimal::Decimal;
use std::str::FromStr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use strategy::{Position, ScalpingStrategy};
use tokio::time;
use url::form_urlencoded::parse;

#[derive(Debug)]
struct TradingBot {
    client: BinanceClient,
    strategy: ScalpingStrategy,
    config: Config,
    is_running: bool,
}

impl TradingBot {
    pub fn new(config: Config) -> Self {
        let client = BinanceClient::new(
            config.binance.api_key.clone(),
            config.binance.secret_key.clone(),
            config.binance.base_url.clone(),
        );

        let strategy = ScalpingStrategy::new(config.trading.clone());

        Self {
            client,
            strategy,
            config,
            is_running: false,
        }
    }

    pub async fn start(&mut self) -> Result<()> {
        info!("Starting Binance Scalping Bot...");

        // Verify API connection
        self.verify_connection().await?;

        self.is_running = true;

        // Handle Ctrl+C gracefully - simplified approach
        let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let running_clone = running.clone();

        ctrlc::set_handler(move || {
            info!("Received Ctrl+C, shutting down gracefully...");
            running_clone.store(false, std::sync::atomic::Ordering::SeqCst);
        })?;

        // Main trading loop
        let mut interval = time::interval(Duration::from_secs(5)); // Check every 5 seconds

        // Initialize strategy with market data7

        self.strategy.update_market_data(&self.client).await?;
        self.strategy
            .update_higher_tf_market_data(&self.client)
            .await?;

        // Initialize ML model if enabled
        if self.config.trading.ml_enabled {
            if let Err(e) = self.strategy.initialize_ml().await {
                error!("Failed to initialize ML model: {e}");
                return Err(e);
            }
        }

        while self.is_running {
            if !running.load(std::sync::atomic::Ordering::SeqCst) {
                info!("Shutting down trading bot...");
                self.is_running = false;
                break;
            }
            interval.tick().await;

            if let Err(e) = self.trading_cycle().await {
                error!("Error in trading cycle: {e}");
                // Continue running but log the error
            }

            // Print performance stats every 5 minutes
            if SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() % 300 == 0 {
                let (total, winning, pnl, win_rate) = self.strategy.get_performance_stats();
                info!(
                    "Stats - Trades: {total}, Wins: {winning}, P&L: {pnl:.4}, Win Rate: {win_rate:.2}%",
                );
            }

            // Print performance stats every 100 cycles
            // let (total, winning, pnl, win_rate) = self.strategy.get_performance_stats();
            // if total > 0 && total % 10 == 0 {
            //     info!(
            //         "Stats - Trades: {total}, Wins: {winning}, P&L: {pnl:.4}, Win Rate: {win_rate:.2}%",
            //     );
            // }
        }

        Ok(())
    }

    async fn verify_connection(&self) -> Result<()> {
        info!("Verifying API connection...");

        let account_info = self.client.get_account_info().await?;
        info!(
            "Connected successfully. Account can trade: {}",
            account_info.can_trade
        );

        if !account_info.can_trade {
            return Err(anyhow::anyhow!("Account is not allowed to trade"));
        }

        // Check if we have enough balance
        let quote_asset = if self.config.trading.symbol.ends_with("USDT") {
            "USDT"
        } else if self.config.trading.symbol.ends_with("BTC") {
            "BTC"
        } else {
            "USDT" // Default
        };

        if let Some(balance) = account_info
            .balances
            .iter()
            .find(|b| b.asset == quote_asset)
        {
            let free_balance = Decimal::from_str(&balance.free)?;
            let required_balance = self.config.trading.position_size
                * Decimal::from(self.config.trading.max_positions);

            if free_balance < required_balance {
                warn!("Low balance. Free: {free_balance}, Required: {required_balance}");
            } else {
                info!("Sufficient balance. Free: {free_balance}");
            }
        } else {
            return Err(anyhow::anyhow!(
                "Could not find balance for asset: {}",
                quote_asset
            ));
        }

        Ok(())
    }

    async fn trading_cycle(&mut self) -> Result<()> {
        // Update market data
        self.strategy.update_market_data(&self.client).await?;
        self.strategy
            .update_higher_tf_market_data(&self.client)
            .await?;

        // Check for exit conditions on existing positions
        if let Some(current_price) = self.get_current_price().await? {
            let positions_to_close = self.strategy.check_exit_conditions(current_price);

            // Close positions that hit targets or stop losses
            for &position_index in &positions_to_close {
                let mut position_ml_train: Option<Position> = None;
                let mut close_price = current_price;
                let mut pnl = Decimal::ZERO;
                let mut profit_pct = Decimal::ZERO;

                if let Some(position) = self.strategy.get_positions().get(position_index) {
                    position_ml_train = Some(position.clone());
                    (close_price, pnl, profit_pct) =
                        self.close_position(position.clone(), current_price).await?;
                }
                if let Some(position_ml) = position_ml_train {
                    if let Err(e) =
                        self.strategy
                            .update_ml_training_data(&position_ml, pnl, profit_pct)
                    {
                        warn!("Failed to update ML training data: {}", e);
                    }
                }
            }

            // Remove closed positions
            self.strategy.remove_positions(positions_to_close);

            let mut signal = Signal::Hold;
            let mut ml_prediction = 0.5;
            let mut features = Vec::new();

            // Check if we are in cooldown period
            let current_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

            // Check cooldown
            if let Some(last_time) = self.strategy.last_signal_time {
                if current_time > last_time + self.strategy.cooldown_duration.as_secs() {
                    // log::info!("In cooldown period, skipping signal generation.");
                    (signal, ml_prediction, features) = self.strategy.analyze_market()?;
                }
            } else {
                // No previous signal, analyze market
                (signal, ml_prediction, features) = self.strategy.analyze_market()?;
            }

            // Analyze market for new opportunities
            match signal {
                Signal::Buy => {
                    info!("BUY signal detected at price: {current_price:.4}");
                    self.strategy.last_signal_time = Some(current_time);
                    self.execute_buy_order(current_price, ml_prediction, features)
                        .await?;
                }
                Signal::Sell => {
                    info!("SELL signal detected at price: {current_price:.4}");
                    self.strategy.last_signal_time = Some(current_time);
                    self.execute_sell_order(current_price, ml_prediction, features)
                        .await?;
                }
                Signal::Hold => {
                    // Do nothing
                }
            }
        }

        Ok(())
    }

    async fn get_current_price(&self) -> Result<Option<Decimal>> {
        let ticker = self
            .client
            .get_order_book_ticker(&self.config.trading.symbol)
            .await?;
        let bid = Decimal::from_str(&ticker.bid_price)?;
        let ask = Decimal::from_str(&ticker.ask_price)?;

        // Use mid price
        let mid_price = (bid + ask) / Decimal::from(2);
        Ok(Some(mid_price))
    }

    async fn execute_buy_order(
        &mut self,
        current_price: Decimal,
        ml_prediction: f64,
        features: Vec<f64>,
    ) -> Result<()> {
        let quantity = self.strategy.calculate_position_size(current_price)?;

        // Format quantity to appropriate precision (this should be based on symbol info)
        let quantity_str = format!("{quantity:.6}");

        info!(
            "Placing BUY order: {} {} at {:.4}",
            quantity_str, self.config.trading.symbol, current_price
        );

        // For testing, we'll use a limit order slightly above current price
        let order_price = current_price * Decimal::from_str("1.001")?; // 0.1% above market
        let price_str = format!("{order_price:.4}");

        let order_response = self
            .client
            .place_limit_order(
                &self.config.trading.symbol,
                "BUY",
                &quantity_str,
                &price_str,
                "GTC", // Good Till Cancelled
            )
            .await?;

        let order_price = Decimal::from_str(&order_response.price)?;

        info!(
            "BUY order placed successfully. Order ID: {}, Price: {}",
            order_response.order_id, order_price
        );

        // Calculate targets
        let (target_price, stop_loss) = self.strategy.calculate_targets(order_price, "BUY");

        // Create position
        let position = Position {
            side: "BUY".to_string(),
            quantity,
            entry_price: order_price,
            target_price,
            stop_loss,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            ml_prediction, // Store prediction from ML model
            features,      // Store features for ML model
        };

        self.strategy.add_position(position);

        info!("Position added - Price: {order_price:.4} Target: {target_price:.4}, Stop Loss: {stop_loss:.4}",);

        Ok(())
    }

    async fn execute_sell_order(
        &mut self,
        current_price: Decimal,
        ml_prediction: f64,
        features: Vec<f64>,
    ) -> Result<()> {
        let quantity = self.strategy.calculate_position_size(current_price)?;

        // Format quantity to appropriate precision
        let quantity_str = format!("{quantity:.6}");

        info!(
            "Placing SELL order: {} {} at {:.4}",
            quantity_str, self.config.trading.symbol, current_price
        );

        // For testing, we'll use a limit order slightly below current price
        let order_price = current_price * Decimal::from_str("0.999")?; // 0.1% below market
        let price_str = format!("{order_price:.4}");

        let order_response = self
            .client
            .place_limit_order(
                &self.config.trading.symbol,
                "SELL",
                &quantity_str,
                &price_str,
                "GTC",
            )
            .await?;

        let order_price = Decimal::from_str(&order_response.price)?;

        info!(
            "SELL order placed successfully. Order ID: {}, Price: {}",
            order_response.order_id, order_price,
        );

        // Calculate targets
        let (target_price, stop_loss) = self.strategy.calculate_targets(order_price, "SELL");

        // Create position
        let position = Position {
            side: "SELL".to_string(),
            quantity,
            entry_price: order_price,
            target_price,
            stop_loss,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            ml_prediction, // Store prediction from ML model
            features,      // Store features for ML model
        };

        self.strategy.add_position(position);

        info!("Position added - Price: {order_price:.4} Target: {target_price:.4}, Stop Loss: {stop_loss:.4}",);

        Ok(())
    }

    async fn close_position(
        &self,
        position: Position,
        current_price: Decimal,
    ) -> Result<(Decimal, Decimal, Decimal)> {
        let quantity_str = format!("{:.6}", position.quantity);
        let _price_str = format!("{current_price:.4}");

        // Determine the closing side (opposite of opening)
        let close_side = match position.side.as_str() {
            "BUY" => "SELL",
            "SELL" => "BUY",
            _ => return Err(anyhow::anyhow!("Unknown position side")),
        };

        info!(
            "Closing {} position: {} {} at {:.4}",
            position.side, quantity_str, self.config.trading.symbol, current_price
        );

        let order_response = self
            .client
            .place_market_order(&self.config.trading.symbol, close_side, &quantity_str)
            .await?;

        info!(
            "Order status: {}; Price: {}",
            order_response.status, order_response.price
        );

        // let current_price = Decimal::from_str(&order_response.price)?;
        let executed_qty = Decimal::from_str(&order_response.executed_qty)?;

        info!(
            "Position closed successfully. Order ID: {}, Close price: {:.4}, Close quantity: {}",
            order_response.order_id, order_response.price, executed_qty,
        );

        // Calculate P&L
        let pnl = match position.side.as_str() {
            "BUY" => (current_price - position.entry_price) * executed_qty,
            "SELL" => (position.entry_price - current_price) * executed_qty,
            _ => Decimal::ZERO,
        };

        // calculate profit percentage
        let profit_pct = pnl / (position.entry_price * executed_qty);

        info!("P&L for this trade: {pnl:.4} USDT",);

        Ok((current_price, pnl, profit_pct))
    }

    pub fn stop(&mut self) {
        info!("Stopping trading bot...");
        self.is_running = false;
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    info!("Initializing Binance Scalping Bot");

    // Load configuration
    let config = Config::load()?;

    // Create and start the bot
    let mut bot = TradingBot::new(config);

    // Handle Ctrl+C gracefully - simplified approach
    // let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    // let running_clone = running.clone();

    // ctrlc::set_handler(move || {
    //     info!("Received Ctrl+C, shutting down gracefully...");
    //     running_clone.store(false, std::sync::atomic::Ordering::SeqCst);
    // })?;

    // Start the bot
    bot.start().await?;
    info!("Trading bot stopped successfully.");
    // Wait for the bot to finish
    time::sleep(Duration::from_secs(1)).await;

    Ok(())
}
