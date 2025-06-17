use crate::binance::{BinanceClient, OrderBookTicker, Ticker24hr};
use crate::config::TradingConfig;
use crate::indicators::*;
use anyhow::Result;
use rust_decimal::Decimal;
use std::collections::VecDeque;
use std::str::FromStr;
use log::info;
use std::time::{SystemTime, UNIX_EPOCH, Duration};

#[derive(Debug, Clone)]
pub struct Position {
    pub side: String,
    pub quantity: Decimal,
    pub entry_price: Decimal,
    pub target_price: Decimal,
    pub stop_loss: Decimal,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct MarketData {
    pub prices: VecDeque<Decimal>,
    pub volumes: VecDeque<Decimal>,
    pub timestamps: VecDeque<u64>,
    pub orderbook: Option<OrderBookTicker>,
    pub ticker_24hr: Option<Ticker24hr>,
}

impl MarketData {
    pub fn new(max_size: usize) -> Self {
        Self {
            prices: VecDeque::with_capacity(max_size),
            volumes: VecDeque::with_capacity(max_size),
            timestamps: VecDeque::with_capacity(max_size),
            orderbook: None,
            ticker_24hr: None,
        }
    }

    pub fn add_price_data(
        &mut self,
        price: Decimal,
        volume: Decimal,
        timestamp: u64,
        max_size: usize,
    ) {
        self.prices.push_back(price);
        self.volumes.push_back(volume);
        self.timestamps.push_back(timestamp);

        while self.prices.len() > max_size {
            self.prices.pop_front();
            self.volumes.pop_front();
            self.timestamps.pop_front();
        }
    }

    pub fn update_orderbook(&mut self, orderbook: OrderBookTicker) {
        self.orderbook = Some(orderbook);
    }

    pub fn update_ticker(&mut self, ticker: Ticker24hr) {
        self.ticker_24hr = Some(ticker);
    }

    pub fn get_latest_price(&self) -> Option<Decimal> {
        self.prices.back().copied()
    }

    pub fn get_spread(&self) -> Option<Decimal> {
        if let Some(ref orderbook) = self.orderbook {
            let bid = Decimal::from_str(&orderbook.bid_price).ok()?;
            let ask = Decimal::from_str(&orderbook.ask_price).ok()?;
            Some(calculate_spread_percentage(bid, ask))
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct ScalpingStrategy {
    config: TradingConfig,
    pub market_data: MarketData,

    // Technical indicators
    ema_crossover: EMACrossover,
    rsi: RSI,
    macd: MACD,
    bollinger_bands: BollingerBands,
    pub volume_profile: VolumeProfile,

    // Higher timeframe indicators
    pub higher_tf_ema_crossover: EMACrossover,
    pub higher_tf_market_data: MarketData,

    // Current positions
    positions: Vec<Position>,

    // Performance tracking
    total_trades: u32,
    winning_trades: u32,
    total_pnl: Decimal,
    consecutive_losses: u32,

    // Cooldown period for trades
    pub last_signal_time: Option<u64>, // Timestamp of the last signal
    pub cooldown_duration: Duration,  // Cooldown period
}

impl ScalpingStrategy {
    pub fn new(config: TradingConfig) -> Self {
        Self {
            config: config.clone(),
            market_data: MarketData::new(200),

            // Initialize indicators with common scalping periods
            ema_crossover: EMACrossover::new(5, 13),
            rsi: RSI::new(14),                          // 14-period RSI
            macd: MACD::new(12, 26, 9), // Standard MACD
            bollinger_bands: BollingerBands::new(20, Decimal::from(2)), // 20-period BB
            volume_profile: VolumeProfile::new(2),

            higher_tf_ema_crossover: EMACrossover::new(50, 200), // 50-period EMA for higher timeframe
            higher_tf_market_data: MarketData::new(200), // Higher timeframe market data

            positions: Vec::new(),
            total_trades: 0,
            winning_trades: 0,
            total_pnl: Decimal::ZERO,
            consecutive_losses: 0,

            last_signal_time: None,
            cooldown_duration: Duration::from_secs(config.cooldown_period as u64),
        }
    }

    pub async fn update_market_data(&mut self, client: &BinanceClient) -> Result<()> {
        // Fill initial market data if empty
        if self.market_data.prices.is_empty() {
            info!("Fetching initial market data...");
            let klines = client
                .get_klines(&self.config.symbol, &self.config.timeframe, 200)
                .await?;
            for kline in klines {
                let close_price = Decimal::from_str(&kline.close)?;
                let volume = Decimal::from_str(&kline.volume)?;
                self.market_data
                    .add_price_data(close_price, volume, kline.close_time, 200);
            }
            info!("Initial market data loaded.");
        }
        // Get latest kline data
        let klines = client
            .get_klines(&self.config.symbol, &self.config.timeframe, 1)
            .await?;
        if let Some(kline) = klines.first() {
            let close_price = Decimal::from_str(&kline.close)?;
            let volume = Decimal::from_str(&kline.volume)?;
            self.market_data
                .add_price_data(close_price, volume, kline.close_time, 200);

            // Update volume profile
            self.volume_profile.add_trade(close_price, volume);
        }

        // Get orderbook data
        let orderbook = client.get_order_book_ticker(&self.config.symbol).await?;
        self.market_data.update_orderbook(orderbook);

        // Get 24hr ticker
        let ticker = client.get_24hr_ticker(&self.config.symbol).await?;
        self.market_data.update_ticker(ticker);

        Ok(())
    }

    pub async fn update_higher_tf_market_data(&mut self, client: &BinanceClient) -> Result<()> {
        // Fill higher timeframe market data
        if self.higher_tf_market_data.prices.is_empty() {
            info!("Fetching initial higher timeframe market data...");
            let klines = client
                .get_klines(&self.config.symbol, "5m", 200) // Example: 1 hour timeframe
                .await?;
            for kline in klines {
                let close_price = Decimal::from_str(&kline.close)?;
                let volume = Decimal::from_str(&kline.volume)?;
                self.higher_tf_market_data
                    .add_price_data(close_price, volume, kline.close_time, 200);
            }
            info!("Initial higher timeframe market data loaded.");
        }
        // Get latest kline data for higher timeframe
        let klines = client
            .get_klines(&self.config.symbol, "5m", 1) // Example: 1 hour timeframe
            .await?;
        if let Some(kline) = klines.first() {
            let close_price = Decimal::from_str(&kline.close)?;
            let volume = Decimal::from_str(&kline.volume)?;
            self.higher_tf_market_data
                .add_price_data(close_price, volume, kline.close_time, 200);
        }

        // Update higher timeframe indicators
        self.higher_tf_ema_crossover.update(self.higher_tf_market_data.get_latest_price().unwrap_or(Decimal::ZERO));

        Ok(())
    }

    pub fn analyze_market(&mut self) -> Result<Signal> {
        let current_price = self
            .market_data
            .get_latest_price()
            .ok_or_else(|| anyhow::anyhow!("No price data available"))?;

        // Update all indicators
        let ema_crossover_signal = self.ema_crossover.update(current_price);
        self.rsi.update(current_price);
        self.macd.update(current_price);
        self.bollinger_bands.update(current_price);

        // Perform pre-trade checks or if we already have max positions
        if !self.should_trade()?
            || self.positions.len() >= self.config.max_positions as usize {
            return Ok(Signal::Hold);
        }

        // Generate signals based on multiple indicators
        let mut buy_signals = 0;
        let mut sell_signals = 0;

        // EMA Crossover Signal
        if let Some(signal) = ema_crossover_signal {
            match signal {
                Signal::Buy => buy_signals += 1,
                Signal::Sell => sell_signals += 1,
                _ => {}
            }
        }

        // RSI Signal (looking for quick reversals)
        match self.rsi.signal() {
            Signal::Buy => buy_signals += 1,
            Signal::Sell => sell_signals += 1,
            _ => {}
        }

        // MACD Signal
        match self.macd.signal() {
            Signal::Buy => buy_signals += 1,
            Signal::Sell => sell_signals += 1,
            _ => {}
        }

        // Bollinger Bands Signal
        match self.bollinger_bands.signal(&current_price) {
            Signal::Buy => buy_signals += 1,
            Signal::Sell => sell_signals += 1,
            _ => {}
        }

        // Volume confirmation
        let volume_confirmation = self.check_volume_confirmation();
        if volume_confirmation {
            buy_signals += 1; // Volume confirmation boosts buy signals
        } else {
            sell_signals += 1; // Lack of volume confirmation boosts sell signals
        }

        // Momentum check
        let momentum = self.calculate_momentum();
        if momentum > Decimal::from_str("0.1")? {
            buy_signals += 1;
        } else if momentum < Decimal::from_str("-0.1")? {
            sell_signals += 1;
        }

        self.analyze_signals(&buy_signals, &sell_signals)
    }

    fn analyze_signals(&mut self, buy_signals: &i32, sell_signals: &i32) -> Result<Signal> {
        let mut filtered_signal = Signal::Hold; // Default to hold

        // Decision logic (require at least 2 signals)
        if buy_signals >= &3 && buy_signals > sell_signals {
            filtered_signal = self.filter_by_trend(Signal::Buy);
        } else if sell_signals >= &3 && sell_signals > buy_signals {
            filtered_signal = self.filter_by_trend(Signal::Sell);
        }

        Ok(filtered_signal)
    }

    fn should_trade(&self) -> Result<bool> {
        // Check a spread threshold
        if let Some(spread) = self.market_data.get_spread() {
            if spread > self.config.spread_threshold {
                log::warn!("Spread too wide: {spread:.4}%");
                return Ok(false);
            }
        }

        // Check minimum volume
        if let Some(ref ticker) = self.market_data.ticker_24hr {
            let volume = Decimal::from_str(&ticker.quote_volume)?;
            if volume < self.config.min_volume {
                log::warn!("Volume too low: {volume}");
                return Ok(false);
            }
        }

        // Check consecutive losses
        if self.consecutive_losses >= self.config.max_positions {
            log::warn!("Too many consecutive losses: {}", self.consecutive_losses);
            return Ok(false);
        }

        Ok(true)
    }

    fn check_volume_confirmation(&self) -> bool {
        if self.market_data.volumes.len() < 2 {
            return false;
        }

        let current_volume = self.market_data.volumes.back().unwrap();
        let avg_volume = self.market_data.volumes.iter().sum::<Decimal>()
            / Decimal::from(self.market_data.volumes.len());

        *current_volume > avg_volume * Decimal::from_str("1.2").unwrap_or(Decimal::ONE)
    }

    fn calculate_momentum(&self) -> Decimal {
        let prices: Vec<Decimal> = self.market_data.prices.iter().copied().collect();
        calculate_price_momentum(&prices, 3) // 3-period momentum
    }

    fn calculate_volatility(&self) -> Decimal {
        let prices: Vec<Decimal> = self.market_data.prices.iter().copied().collect();
        calculate_volatility(&prices, 20) // 20-period volatility
    }

    pub fn calculate_position_size(&self, current_price: Decimal) -> Result<Decimal> {
        // Simple position sizing based on fixed dollar amount
        let position_value = self.config.position_size;
        let quantity = position_value / current_price;

        // Round to appropriate precision (this should be based on symbol info)
        Ok(quantity)
    }

    pub fn calculate_targets(&self, entry_price: Decimal, side: &str) -> (Decimal, Decimal) {
        // Calculate dynamic stop loss based on volatility
        let volatility = self.calculate_volatility(); // Example: Bollinger Bands width
        let dynamic_stop_pct = self.config.stop_loss_pct + (volatility / Decimal::from(100));

        // Calculate dynamic take profit based on momentum
        let momentum = self.calculate_momentum(); // Example: Price momentum
        let dynamic_target_pct = self.config.scalp_target_pct + (momentum / Decimal::from(100));

        let target_pct = if self.config.dynamic_targets {dynamic_target_pct} else { self.config.scalp_target_pct };
        let stop_pct = if self.config.dynamic_targets {dynamic_stop_pct} else {self.config.stop_loss_pct};

        match side {
            "BUY" => {
                let target = entry_price * (Decimal::ONE + target_pct);
                let stop_loss = entry_price * (Decimal::ONE - stop_pct);
                (target, stop_loss)
            }
            "SELL" => {
                let target = entry_price * (Decimal::ONE - target_pct);
                let stop_loss = entry_price * (Decimal::ONE + stop_pct);
                (target, stop_loss)
            }
            _ => (entry_price, entry_price),
        }
    }

    pub fn add_position(&mut self, position: Position) {
        self.positions.push(position);
        self.total_trades += 1;
    }

    pub fn check_exit_conditions(&mut self, current_price: Decimal) -> Vec<usize> {
        let mut positions_to_close = Vec::new();

        for (i, position) in self.positions.iter().enumerate() {
            let should_exit = match position.side.as_str() {
                "BUY" => {
                    current_price >= position.target_price || current_price <= position.stop_loss
                }
                "SELL" => {
                    current_price <= position.target_price || current_price >= position.stop_loss
                }
                _ => false,
            };

            if should_exit {
                positions_to_close.push(i);

                // Update P&L tracking
                let pnl = self.calculate_pnl(position, current_price);
                self.total_pnl += pnl;

                if pnl > Decimal::ZERO {
                    self.winning_trades += 1;
                    self.consecutive_losses = 0;
                } else {
                    self.consecutive_losses += 1;
                }
            }
        }

        positions_to_close
    }

    fn calculate_pnl(&self, position: &Position, exit_price: Decimal) -> Decimal {
        match position.side.as_str() {
            "BUY" => (exit_price - position.entry_price) * position.quantity,
            "SELL" => (position.entry_price - exit_price) * position.quantity,
            _ => Decimal::ZERO,
        }
    }

    pub fn remove_positions(&mut self, indices: Vec<usize>) {
        // Sort in reverse order to avoid index shifting
        let mut sorted_indices = indices;
        sorted_indices.sort_by(|a, b| b.cmp(a));

        for index in sorted_indices {
            if index < self.positions.len() {
                self.positions.remove(index);
            }
        }
    }

    pub fn get_positions(&self) -> &Vec<Position> {
        &self.positions
    }

    pub fn get_performance_stats(&self) -> (u32, u32, Decimal, f64) {
        let win_rate = if self.total_trades > 0 {
            (self.winning_trades as f64 / self.total_trades as f64) * 100.0
        } else {
            0.0
        };

        (
            self.total_trades,
            self.winning_trades,
            self.total_pnl,
            win_rate,
        )
    }

    // Returns Some("up"), Some("down"), or None if no clear trend
    pub fn get_trend(&self) -> Option<&'static str> {

        match self.ema_crossover.signal() {
            Signal::Buy => Some("up"),
            Signal::Sell => Some("down"),
            _ => None,
        }
    }

    // Example: Only allow buys in uptrend, sells in downtrend
    pub fn filter_by_trend(&self, signal: Signal) -> Signal {
        match (self.get_trend(), signal) {
            (Some("up"), Signal::Buy) => {
                info!("Uptrend detected, allowing buy signal");
                Signal::Buy
            },
            (Some("down"), Signal::Sell) => {
                info!("Downtrend detected, allowing sell signal");
                Signal::Sell
            },
            _ => {
                info!("No clear trend, holding position");
                Signal::Hold
            },
        }
    }
}
