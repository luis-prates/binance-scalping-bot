use crate::binance::{BinanceClient, Kline};
use crate::config::TradingConfig;
use crate::indicators::Signal;
use crate::strategy::{Position, ScalpingStrategy};
use anyhow::Result;
use chrono::{DateTime, NaiveDateTime, Utc};
use log::info;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub start_date: String, // "2024-01-01"
    pub end_date: String,   // "2024-12-31"
    pub initial_balance: Decimal,
    pub commission_rate: Decimal,     // 0.001 = 0.1%
    pub slippage: Decimal,            // 0.0001 = 0.01%
    pub data_interval: String,        // "1m", "5m", etc.
    pub max_klines_per_request: u16,  // 1000
    pub filter_data_interval: String, // "1m", "5m", etc.
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            start_date: "2024-01-01".to_string(),
            end_date: "2024-01-31".to_string(),
            initial_balance: Decimal::from(1000),
            commission_rate: Decimal::from_str("0.001").unwrap(), // 0.1%
            slippage: Decimal::from_str("0.0001").unwrap(),       // 0.01%
            data_interval: "1m".to_string(),
            max_klines_per_request: 1000,
            filter_data_interval: "5m".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BacktestTrade {
    pub id: u64,
    pub symbol: String,
    pub side: String,
    pub entry_time: DateTime<Utc>,
    pub exit_time: Option<DateTime<Utc>>,
    pub entry_price: Decimal,
    pub exit_price: Option<Decimal>,
    pub quantity: Decimal,
    pub commission: Decimal,
    pub pnl: Option<Decimal>,
    pub exit_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BacktestResults {
    pub initial_balance: Decimal,
    pub final_balance: Decimal,
    pub total_return: Decimal,
    pub total_return_pct: Decimal,
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub win_rate: f64,
    pub avg_win: Decimal,
    pub avg_loss: Decimal,
    pub largest_win: Decimal,
    pub largest_loss: Decimal,
    pub max_drawdown: Decimal,
    pub max_drawdown_pct: Decimal,
    pub sharpe_ratio: f64,
    pub profit_factor: Decimal,
    pub total_commission: Decimal,
    pub trades: Vec<BacktestTrade>,
    pub daily_returns: Vec<(DateTime<Utc>, Decimal)>,
    pub equity_curve: Vec<(DateTime<Utc>, Decimal)>,
}

#[derive(Debug)]
pub struct Backtester {
    client: BinanceClient,
    config: BacktestConfig,
    trading_config: TradingConfig,
    balance: Decimal,
    equity_curve: Vec<(DateTime<Utc>, Decimal)>,
    trades: Vec<BacktestTrade>,
    open_positions: HashMap<u64, Position>,
    next_trade_id: u64,
    total_commission: Decimal,
}

impl Backtester {
    pub fn new(
        client: BinanceClient,
        config: BacktestConfig,
        trading_config: TradingConfig,
    ) -> Self {
        let initial_balance = config.initial_balance;

        Self {
            client,
            config,
            trading_config,
            balance: initial_balance,
            equity_curve: vec![(Utc::now(), initial_balance)],
            trades: Vec::new(),
            open_positions: HashMap::new(),
            next_trade_id: 1,
            total_commission: Decimal::ZERO,
        }
    }

    pub async fn run_backtest(&mut self) -> Result<BacktestResults> {
        log::info!(
            "Starting backtest from {} to {}",
            self.config.start_date,
            self.config.end_date
        );

        // Get historical data
        let klines = self.fetch_historical_data().await?;
        log::info!("Loaded {} klines for backtesting", klines.len());

        if klines.is_empty() {
            return Err(anyhow::anyhow!(
                "No historical data available for the specified period"
            ));
        }

        // Initialize strategy
        let mut strategy = ScalpingStrategy::new(self.trading_config.clone());

        let mut htf_buffer = Vec::new();
        // Convert filter_data_interval to milliseconds from 5m
        let htf_interval_ms = match self.config.filter_data_interval.as_str() {
            "1m" => 60_000,
            "5m" => 300_000,
            "15m" => 900_000,
            "30m" => 1_800_000,
            "1h" => 3_600_000,
            "4h" => 14_400_000,
            "1d" => 86_400_000,
            _ => return Err(anyhow::anyhow!("Unsupported filter data interval")),
        };

        info!(
            "Using higher timeframe filter interval of {}ms",
            htf_interval_ms
        );

        // Process each kline
        for (i, kline) in klines.iter().enumerate() {
            self.process_kline(&mut strategy, kline, i, &mut htf_buffer, htf_interval_ms)
                .await?;

            // Log progress every 1000 klines
            if i % 1000 == 0 {
                log::info!("Processed {} / {} klines", i + 1, klines.len());
            }
        }

        // Close any remaining open positions at final price
        if let Some(final_kline) = klines.last() {
            let final_price = Decimal::from_str(&final_kline.close)?;
            self.close_all_positions(final_price, "Backtest ended")
                .await?;
        }

        // Generate results
        let results = self.generate_results().await?;

        log::info!(
            "Backtest completed. Final balance: {:.2}",
            results.final_balance
        );
        log::info!("Total return: {:.2}%", results.total_return_pct);
        log::info!("Win rate: {:.2}%", results.win_rate);

        Ok(results)
    }

    async fn fetch_historical_data(&self) -> Result<Vec<Kline>> {
        let start_time = self.parse_date(&self.config.start_date)?;
        let end_time = self.parse_date(&self.config.end_date)?;

        let mut all_klines = Vec::new();
        let mut current_time = start_time;

        while current_time < end_time {
            // Calculate the end time for this batch
            let batch_end = std::cmp::min(
                current_time + chrono::Duration::minutes(self.config.max_klines_per_request as i64),
                end_time,
            );

            log::info!(
                "Fetching data from {} to {}",
                current_time.format("%Y-%m-%d %H:%M:%S"),
                batch_end.format("%Y-%m-%d %H:%M:%S")
            );

            // Fetch klines for this batch
            let klines = self
                .client
                .get_klines_with_range(
                    &self.trading_config.symbol,
                    &self.config.data_interval,
                    Some(current_time.timestamp_millis() as u64),
                    Some(batch_end.timestamp_millis() as u64),
                    self.config.max_klines_per_request,
                )
                .await?;

            if klines.is_empty() {
                break;
            }

            all_klines.extend(klines);

            // Move to next batch
            current_time = batch_end;

            // Add small delay to avoid rate limits
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        // Sort by open time to ensure chronological order
        all_klines.sort_by(|a, b| a.open_time.cmp(&b.open_time));

        Ok(all_klines)
    }

    async fn process_kline(
        &mut self,
        strategy: &mut ScalpingStrategy,
        kline: &Kline,
        _index: usize,
        htf_buffer: &mut Vec<Kline>,
        htf_interval_ms: u64,
    ) -> Result<()> {
        let current_price = Decimal::from_str(&kline.close)?;
        let current_time = DateTime::from_timestamp(
            (kline.close_time / 1000) as i64,
            ((kline.close_time % 1000) * 1_000_000) as u32,
        )
        .unwrap_or_else(|| Utc::now());

        // Update strategy with current kline data
        strategy.update_market_data_from_kline(kline)?;

        htf_buffer.push(kline.clone());
        if (kline.close_time + 1) % htf_interval_ms == 0 {
            // info!("Processing higher timeframe candle at open: {}, close: {}", kline.open_time, kline.close_time + 1);
            // Aggregate OHLCV for the 5m candle
            let open = &htf_buffer.first().unwrap().open;
            let close = &htf_buffer.last().unwrap().close;
            let high = htf_buffer
                .iter()
                .map(|k| k.high.parse::<Decimal>().unwrap())
                .max()
                .unwrap();
            let low = htf_buffer
                .iter()
                .map(|k| k.low.parse::<Decimal>().unwrap())
                .min()
                .unwrap();
            let volume: Decimal = htf_buffer
                .iter()
                .map(|k| k.volume.parse::<Decimal>().unwrap())
                .sum();

            // Create a synthetic 5m kline and update higher timeframe indicators
            let htf_kline = Kline {
                open_time: htf_buffer.first().unwrap().open_time,
                open: open.clone(),
                high: high.to_string(),
                low: low.to_string(),
                close: close.clone(),
                volume: volume.to_string(),
                close_time: htf_buffer.last().unwrap().close_time,
                // ... fill other fields as needed
                ..htf_buffer.last().unwrap().clone()
            };
            strategy.higher_tf_market_data.add_price_data(
                Decimal::from_str(&htf_kline.close)?,
                Decimal::from_str(&htf_kline.volume)?,
                htf_kline.close_time,
                200,
            );
            strategy
                .higher_tf_ema_crossover
                .update(Decimal::from_str(&htf_kline.close)?);

            htf_buffer.clear();
        }

        // Check exit conditions for open positions
        self.check_exit_conditions(strategy, current_price, current_time)
            .await?;

        // Generate new signals
        let (signal, _, _) = strategy.analyze_market()?;

        match signal {
            Signal::Buy => {
                self.execute_backtest_order("BUY", current_price, current_time, strategy)
                    .await?;
            }
            Signal::Sell => {
                self.execute_backtest_order("SELL", current_price, current_time, strategy)
                    .await?;
            }
            Signal::Hold => {
                // Do nothing
            }
        }

        // Update equity curve
        let current_equity = self.calculate_current_equity(current_price);
        self.equity_curve.push((current_time, current_equity));

        Ok(())
    }

    async fn execute_backtest_order(
        &mut self,
        side: &str,
        price: Decimal,
        time: DateTime<Utc>,
        strategy: &mut ScalpingStrategy,
    ) -> Result<()> {
        // Check if we have reached max positions
        if self.open_positions.len() >= self.trading_config.max_positions as usize {
            return Ok(());
        }

        // Calculate position size
        let quantity = strategy.calculate_position_size(price)?;
        let position_value = quantity * price;

        // Apply slippage
        let execution_price = match side {
            "BUY" => price * (Decimal::ONE + self.config.slippage),
            "SELL" => price * (Decimal::ONE - self.config.slippage),
            _ => price,
        };

        // Calculate commission
        let commission = position_value * self.config.commission_rate;

        // Check if we have enough balance
        let required_balance = position_value + commission;
        if self.balance < required_balance {
            log::warn!(
                "Insufficient balance for trade. Required: {:.2}, Available: {:.2}",
                required_balance,
                self.balance
            );
            return Ok(());
        }

        // Deduct balance
        self.balance -= required_balance;
        self.total_commission += commission;

        // Calculate targets
        let (target_price, stop_loss) = strategy.calculate_targets(execution_price, side);

        // Create position
        let position = Position {
            side: side.to_string(),
            quantity,
            entry_price: execution_price,
            target_price,
            stop_loss,
            timestamp: time.timestamp() as u64,
            features: vec![],
            ml_prediction: 0.0,
        };

        // Create trade record
        let trade = BacktestTrade {
            id: self.next_trade_id,
            symbol: self.trading_config.symbol.clone(),
            side: side.to_string(),
            entry_time: time,
            exit_time: None,
            entry_price: execution_price,
            exit_price: None,
            quantity,
            commission,
            pnl: None,
            exit_reason: None,
        };

        // Store position and trade
        self.open_positions.insert(self.next_trade_id, position);
        self.trades.push(trade);

        log::debug!(
            "Opened {} position #{}: {:.6} @ {:.4}, Target: {:.4}, Stop: {:.4}",
            side,
            self.next_trade_id,
            quantity,
            execution_price,
            target_price,
            stop_loss
        );

        self.next_trade_id += 1;

        Ok(())
    }

    async fn check_exit_conditions(
        &mut self,
        _strategy: &mut ScalpingStrategy,
        current_price: Decimal,
        current_time: DateTime<Utc>,
    ) -> Result<()> {
        let mut positions_to_close = Vec::new();

        for (&trade_id, position) in &self.open_positions {
            let should_exit = match position.side.as_str() {
                "BUY" => {
                    if current_price >= position.target_price {
                        Some(("Profit target", position.target_price))
                    } else if current_price <= position.stop_loss {
                        Some(("Stop loss", position.stop_loss))
                    } else {
                        None
                    }
                }
                "SELL" => {
                    if current_price <= position.target_price {
                        Some(("Profit target", position.target_price))
                    } else if current_price >= position.stop_loss {
                        Some(("Stop loss", position.stop_loss))
                    } else {
                        None
                    }
                }
                _ => None,
            };

            if let Some((reason, exit_price)) = should_exit {
                positions_to_close.push((trade_id, exit_price, reason.to_string()));
            }
        }

        // Close positions that hit targets or stops
        for (trade_id, exit_price, reason) in positions_to_close {
            self.close_position(trade_id, exit_price, current_time, &reason)
                .await?;
        }

        Ok(())
    }

    async fn close_position(
        &mut self,
        trade_id: u64,
        exit_price: Decimal,
        exit_time: DateTime<Utc>,
        exit_reason: &str,
    ) -> Result<()> {
        if let Some(position) = self.open_positions.remove(&trade_id) {
            // Apply slippage
            let execution_price = match position.side.as_str() {
                "BUY" => exit_price * (Decimal::ONE - self.config.slippage), // Selling
                "SELL" => exit_price * (Decimal::ONE + self.config.slippage), // Buying back
                _ => exit_price,
            };

            // Calculate P&L
            let pnl = match position.side.as_str() {
                "BUY" => (execution_price - position.entry_price) * position.quantity,
                "SELL" => (position.entry_price - execution_price) * position.quantity,
                _ => Decimal::ZERO,
            };

            // Calculate commission for closing
            let position_value = position.quantity * execution_price;
            let commission = position_value * self.config.commission_rate;
            let net_pnl = pnl - commission;

            // Add proceeds back to balance
            self.balance += position_value - commission;
            self.total_commission += commission;

            // Update the trade record
            if let Some(trade) = self.trades.iter_mut().find(|t| t.id == trade_id) {
                trade.exit_time = Some(exit_time);
                trade.exit_price = Some(execution_price);
                trade.pnl = Some(net_pnl);
                trade.exit_reason = Some(exit_reason.to_string());
                trade.commission += commission;
            }

            log::debug!(
                "Closed {} position #{}: P&L: {:.4} ({})",
                position.side,
                trade_id,
                net_pnl,
                exit_reason
            );
        }

        Ok(())
    }

    async fn close_all_positions(&mut self, final_price: Decimal, reason: &str) -> Result<()> {
        let position_ids: Vec<u64> = self.open_positions.keys().copied().collect();
        let final_time = Utc::now();

        for trade_id in position_ids {
            self.close_position(trade_id, final_price, final_time, reason)
                .await?;
        }

        Ok(())
    }

    fn calculate_current_equity(&self, current_price: Decimal) -> Decimal {
        let mut equity = self.balance;

        // Add unrealized P&L from open positions
        for position in self.open_positions.values() {
            let unrealized_pnl = match position.side.as_str() {
                "BUY" => (current_price - position.entry_price) * position.quantity,
                "SELL" => (position.entry_price - current_price) * position.quantity,
                _ => Decimal::ZERO,
            };
            equity += unrealized_pnl;
        }

        equity
    }

    async fn generate_results(&self) -> Result<BacktestResults> {
        let final_balance = self.balance;
        let initial_balance = self.config.initial_balance;

        let total_return = final_balance - initial_balance;
        let total_return_pct = if initial_balance > Decimal::ZERO {
            (total_return / initial_balance) * Decimal::from(100)
        } else {
            Decimal::ZERO
        };

        let completed_trades: Vec<&BacktestTrade> =
            self.trades.iter().filter(|t| t.pnl.is_some()).collect();

        let total_trades = completed_trades.len() as u32;
        let mut winning_trades = 0;
        let mut losing_trades = 0;
        let mut total_wins = Decimal::ZERO;
        let mut total_losses = Decimal::ZERO;
        let mut largest_win = Decimal::ZERO;
        let mut largest_loss = Decimal::ZERO;

        for trade in &completed_trades {
            if let Some(pnl) = trade.pnl {
                if pnl > Decimal::ZERO {
                    winning_trades += 1;
                    total_wins += pnl;
                    if pnl > largest_win {
                        largest_win = pnl;
                    }
                } else {
                    losing_trades += 1;
                    total_losses += pnl.abs();
                    if pnl < largest_loss {
                        largest_loss = pnl;
                    }
                }
            }
        }

        let win_rate = if total_trades > 0 {
            (winning_trades as f64 / total_trades as f64) * 100.0
        } else {
            0.0
        };

        let avg_win = if winning_trades > 0 {
            total_wins / Decimal::from(winning_trades)
        } else {
            Decimal::ZERO
        };

        let avg_loss = if losing_trades > 0 {
            total_losses / Decimal::from(losing_trades)
        } else {
            Decimal::ZERO
        };

        let profit_factor = if total_losses > Decimal::ZERO {
            total_wins / total_losses
        } else if total_wins > Decimal::ZERO {
            Decimal::from(999) // Very high profit factor
        } else {
            Decimal::ZERO
        };

        // Calculate max drawdown
        let (max_drawdown, max_drawdown_pct) = self.calculate_max_drawdown();

        // Calculate Sharpe ratio (simplified)
        let sharpe_ratio = self.calculate_sharpe_ratio();

        // Generate daily returns
        let daily_returns = self.calculate_daily_returns();

        Ok(BacktestResults {
            initial_balance,
            final_balance,
            total_return,
            total_return_pct,
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            avg_win,
            avg_loss,
            largest_win,
            largest_loss,
            max_drawdown,
            max_drawdown_pct,
            sharpe_ratio,
            profit_factor,
            total_commission: self.total_commission,
            trades: self.trades.clone(),
            daily_returns,
            equity_curve: self.equity_curve.clone(),
        })
    }

    fn calculate_max_drawdown(&self) -> (Decimal, Decimal) {
        let mut max_equity = self.config.initial_balance;
        let mut max_drawdown = Decimal::ZERO;
        let mut max_drawdown_pct = Decimal::ZERO;

        for &(_, equity) in &self.equity_curve {
            if equity > max_equity {
                max_equity = equity;
            }

            let drawdown = max_equity - equity;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
                max_drawdown_pct = if max_equity > Decimal::ZERO {
                    (drawdown / max_equity) * Decimal::from(100)
                } else {
                    Decimal::ZERO
                };
            }
        }

        (max_drawdown, max_drawdown_pct)
    }

    fn calculate_sharpe_ratio(&self) -> f64 {
        if self.equity_curve.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self
            .equity_curve
            .windows(2)
            .map(|window| {
                let prev = window[0].1;
                let curr = window[1].1;
                if prev > Decimal::ZERO {
                    ((curr - prev) / prev)
                        .to_string()
                        .parse::<f64>()
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            })
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;

        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            mean_return / std_dev * (252.0_f64).sqrt() // Annualized Sharpe
        } else {
            0.0
        }
    }

    fn calculate_daily_returns(&self) -> Vec<(DateTime<Utc>, Decimal)> {
        let mut daily_returns = Vec::new();
        let mut daily_equity: HashMap<String, Decimal> = HashMap::new();

        // Group equity by date
        for &(timestamp, equity) in &self.equity_curve {
            let date_key = timestamp.format("%Y-%m-%d").to_string();
            daily_equity.insert(date_key, equity);
        }

        // Convert to sorted daily returns
        let mut sorted_dates: Vec<_> = daily_equity.keys().collect();
        sorted_dates.sort();

        for (i, date_str) in sorted_dates.iter().enumerate() {
            if let (Some(current_equity), Some(prev_date)) = (
                daily_equity.get(*date_str),
                sorted_dates.get(i.saturating_sub(1)),
            ) {
                if i > 0 {
                    if let Some(prev_equity) = daily_equity.get(*prev_date) {
                        let daily_return = if *prev_equity > Decimal::ZERO {
                            (*current_equity - *prev_equity) / *prev_equity
                        } else {
                            Decimal::ZERO
                        };

                        if let Ok(date) = DateTime::parse_from_str(
                            &format!("{} 00:00:00 +0000", date_str),
                            "%Y-%m-%d %H:%M:%S %z",
                        ) {
                            daily_returns.push((date.with_timezone(&Utc), daily_return));
                        }
                    }
                }
            }
        }

        daily_returns
    }

    fn parse_date(&self, date_str: &str) -> Result<DateTime<Utc>> {
        let naive_date =
            NaiveDateTime::parse_from_str(&format!("{} 00:00:00", date_str), "%Y-%m-%d %H:%M:%S")?;
        Ok(DateTime::from_naive_utc_and_offset(naive_date, Utc))
    }
}

// Add method to strategy for backtest data feeding
impl crate::strategy::ScalpingStrategy {
    pub fn update_market_data_from_kline(&mut self, kline: &Kline) -> Result<()> {
        let close_price = Decimal::from_str(&kline.close)?;
        let volume = Decimal::from_str(&kline.volume)?;

        self.market_data
            .add_price_data(close_price, volume, kline.close_time, 200);
        self.volume_profile.add_trade(close_price, volume);

        Ok(())
    }
}

// Extend BinanceClient for historical data with date range
impl crate::binance::BinanceClient {
    pub async fn get_klines_with_range(
        &self,
        symbol: &str,
        interval: &str,
        start_time: Option<u64>,
        end_time: Option<u64>,
        limit: u16,
    ) -> Result<Vec<Kline>> {
        let mut url = format!(
            "{}/api/v3/klines?symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        if let Some(start) = start_time {
            url.push_str(&format!("&startTime={}", start));
        }

        if let Some(end) = end_time {
            url.push_str(&format!("&endTime={}", end));
        }

        println!("String to post: {}", url);

        let response: Vec<Vec<serde_json::Value>> =
            self.client.get(&url).send().await?.json().await?;

        let klines = response
            .into_iter()
            .map(|k| Kline {
                open_time: k[0].as_u64().unwrap(),
                open: k[1].as_str().unwrap().to_string(),
                high: k[2].as_str().unwrap().to_string(),
                low: k[3].as_str().unwrap().to_string(),
                close: k[4].as_str().unwrap().to_string(),
                volume: k[5].as_str().unwrap().to_string(),
                close_time: k[6].as_u64().unwrap(),
                quote_asset_volume: k[7].as_str().unwrap().to_string(),
                number_of_trades: k[8].as_u64().unwrap() as u32,
                taker_buy_base_asset_volume: k[9].as_str().unwrap().to_string(),
                taker_buy_quote_asset_volume: k[10].as_str().unwrap().to_string(),
            })
            .collect();

        Ok(klines)
    }
}
