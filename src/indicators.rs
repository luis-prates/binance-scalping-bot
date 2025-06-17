use rust_decimal::Decimal;
use std::collections::VecDeque;

// Helper function to calculate the square root of Decimal
fn decimal_sqrt(value: Decimal) -> Decimal {
    if value <= Decimal::ZERO {
        return Decimal::ZERO;
    }
    
    // Use Newton's method for approximation
    let mut x = value / Decimal::from(2);
    let mut prev_x = value;
    
    // Iterate until we have good precision
    for _ in 0..20 {
        if (x - prev_x).abs() < Decimal::new(1, 8) { // Precision to 8 decimal places
            break;
        }
        prev_x = x;
        x = (x + value / x) / Decimal::from(2);
    }
    
    x
}

#[derive(Debug, Clone, PartialEq)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

#[derive(Debug, Clone)]
pub struct MovingAverage {
    period: usize,
    values: VecDeque<Decimal>,
    sum: Decimal,
}

impl MovingAverage {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            values: VecDeque::new(),
            sum: Decimal::ZERO,
        }
    }

    pub fn update(&mut self, value: Decimal) -> Option<Decimal> {
        self.values.push_back(value);
        self.sum += value;

        if self.values.len() > self.period {
            if let Some(old_value) = self.values.pop_front() {
                self.sum -= old_value;
            }
        }

        if self.values.len() == self.period {
            Some(self.sum / Decimal::from(self.period))
        } else {
            None
        }
    }

    pub fn current(&self) -> Option<Decimal> {
        if self.values.len() == self.period {
            Some(self.sum / Decimal::from(self.period))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExponentialMovingAverage {
    period: usize,
    multiplier: Decimal,
    current_value: Option<Decimal>,
}

impl ExponentialMovingAverage {
    pub fn new(period: usize) -> Self {
        let multiplier = Decimal::from(2) / (Decimal::from(period) + Decimal::ONE);
        Self {
            period,
            multiplier,
            current_value: None,
        }
    }

    pub fn update(&mut self, value: Decimal) -> Option<Decimal> {
        if let Some(current) = self.current_value {
            self.current_value = Some((value * self.multiplier) + (current * (Decimal::ONE - self.multiplier)));
        } else {
            self.current_value = Some(value);
        }
        self.current_value
    }

    pub fn current(&self) -> Option<Decimal> {
        self.current_value
    }
}

#[derive(Debug, Clone)]
pub struct EMACrossover {
    fast_ema: ExponentialMovingAverage,
    slow_ema: ExponentialMovingAverage,
    crossover_signal: Option<Signal>,
}

impl EMACrossover {
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        Self {
            fast_ema: ExponentialMovingAverage::new(fast_period),
            slow_ema: ExponentialMovingAverage::new(slow_period),
            crossover_signal: None,
        }
    }

    pub fn update(&mut self, close: Decimal) -> Option<Signal> {
        let fast = self.fast_ema.update(close)?;
        let slow = self.slow_ema.update(close)?;

        if fast > slow && self.crossover_signal != Some(Signal::Buy) {
            self.crossover_signal = Some(Signal::Buy);
            return Some(Signal::Buy);
        } else if fast < slow && self.crossover_signal != Some(Signal::Sell) {
            self.crossover_signal = Some(Signal::Sell);
            return Some(Signal::Sell);
        }

        None
    }

    pub fn current(&self) -> Option<(Decimal, Decimal)> {
        Some((self.fast_ema.current()?, self.slow_ema.current()?))
    }

    pub fn signal(&self) -> Signal {
        self.crossover_signal.clone().unwrap_or(Signal::Hold)
    }
}

#[derive(Debug, Clone)]
pub struct RSI {
    period: usize,
    gains: VecDeque<Decimal>,
    losses: VecDeque<Decimal>,
    avg_gain: Decimal,
    avg_loss: Decimal,
    previous_close: Option<Decimal>,
}

impl RSI {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            gains: VecDeque::new(),
            losses: VecDeque::new(),
            avg_gain: Decimal::ZERO,
            avg_loss: Decimal::ZERO,
            previous_close: None,
        }
    }

    pub fn update(&mut self, close: Decimal) -> Option<Decimal> {
        if let Some(prev_close) = self.previous_close {
            let change = close - prev_close;
            let gain = if change > Decimal::ZERO { change } else { Decimal::ZERO };
            let loss = if change < Decimal::ZERO { -change } else { Decimal::ZERO };

            self.gains.push_back(gain);
            self.losses.push_back(loss);

            if self.gains.len() > self.period {
                self.gains.pop_front();
                self.losses.pop_front();
            }

            if self.gains.len() == self.period {
                self.avg_gain = self.gains.iter().sum::<Decimal>() / Decimal::from(self.period);
                self.avg_loss = self.losses.iter().sum::<Decimal>() / Decimal::from(self.period);

                if self.avg_loss == Decimal::ZERO {
                    return Some(Decimal::from(100));
                }

                let rs = self.avg_gain / self.avg_loss;
                let rsi = Decimal::from(100) - (Decimal::from(100) / (Decimal::ONE + rs));
                self.previous_close = Some(close);
                return Some(rsi);
            }
        }
        self.previous_close = Some(close);
        None
    }

    pub fn current(&self) -> Option<Decimal> {
        if self.gains.len() == self.period && self.avg_loss != Decimal::ZERO {
            let rs = self.avg_gain / self.avg_loss;
            let rsi = Decimal::from(100) - (Decimal::from(100) / (Decimal::ONE + rs));
            Some(rsi)
        } else if self.gains.len() == self.period {
            Some(Decimal::from(100))
        } else {
            None
        }
    }

    pub fn signal(&self) -> Signal {
        if let Some(rsi) = self.current() {
            if rsi > Decimal::from(70) {
                Signal::Sell
            } else if rsi < Decimal::from(30) {
                Signal::Buy
            } else {
                Signal::Hold
            }
        } else {
            Signal::Hold
        }
    }
}

#[derive(Debug, Clone)]
pub struct MACD {
    fast_ema: ExponentialMovingAverage,
    slow_ema: ExponentialMovingAverage,
    signal_ema: ExponentialMovingAverage,
    macd_line: Option<Decimal>,
}

impl MACD {
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast_ema: ExponentialMovingAverage::new(fast_period),
            slow_ema: ExponentialMovingAverage::new(slow_period),
            signal_ema: ExponentialMovingAverage::new(signal_period),
            macd_line: None,
        }
    }

    pub fn update(&mut self, close: Decimal) -> Option<(Decimal, Decimal, Decimal)> {
        let fast = self.fast_ema.update(close)?;
        let slow = self.slow_ema.update(close)?;
        
        self.macd_line = Some(fast - slow);
        let signal = self.signal_ema.update(self.macd_line.unwrap())?;
        let histogram = self.macd_line.unwrap() - signal;
        
        Some((self.macd_line.unwrap(), signal, histogram))
    }

    pub fn current(&self) -> Option<(Decimal, Decimal, Decimal)> {
        let macd = self.macd_line?;
        let signal = self.signal_ema.current()?;
        let histogram = macd - signal;
        Some((macd, signal, histogram))
    }

    pub fn signal(&self) -> Signal {
        if let Some((macd_line, signal_line, histogram)) = self.current() {
            if macd_line > signal_line && histogram > Decimal::ZERO {
                Signal::Buy
            } else if macd_line < signal_line && histogram < Decimal::ZERO {
                Signal::Sell
            } else {
                Signal::Hold
            }
        } else {
            Signal::Hold
        }
    }
}

#[derive(Debug, Clone)]
pub struct BollingerBands {
    sma: MovingAverage,
    values: VecDeque<Decimal>,
    period: usize,
    std_dev_multiplier: Decimal,
}

impl BollingerBands {
    pub fn new(period: usize, std_dev_multiplier: Decimal) -> Self {
        Self {
            sma: MovingAverage::new(period),
            values: VecDeque::new(),
            period,
            std_dev_multiplier,
        }
    }

    pub fn update(&mut self, close: Decimal) -> Option<(Decimal, Decimal, Decimal)> {
        self.values.push_back(close);
        if self.values.len() > self.period {
            self.values.pop_front();
        }

        let middle = self.sma.update(close)?;
        
        if self.values.len() == self.period {
            let variance = self.values
                .iter()
                .map(|v| (*v - middle) * (*v - middle))
                .sum::<Decimal>() / Decimal::from(self.period);
            
            let std_dev = decimal_sqrt(variance);
            let upper = middle + (std_dev * self.std_dev_multiplier);
            let lower = middle - (std_dev * self.std_dev_multiplier);
            
            Some((upper, middle, lower))
        } else {
            None
        }
    }

    pub fn current(&self) -> Option<(Decimal, Decimal, Decimal)> {
        if self.values.len() < self.period {
            return None;
        }

        let middle = self.sma.current()?;
        let variance = self.values
            .iter()
            .map(|v| (*v - middle) * (*v - middle))
            .sum::<Decimal>() / Decimal::from(self.period);

        let std_dev = decimal_sqrt(variance);
        let upper = middle + (std_dev * self.std_dev_multiplier);
        let lower = middle - (std_dev * self.std_dev_multiplier);

        Some((upper, middle, lower))
    }

    pub fn signal(&self, close: &Decimal) -> Signal {
        if let Some((upper, _, lower)) = self.current() {
            if close > &upper {
                Signal::Sell
            } else if close < &lower {
                Signal::Buy
            } else {
                Signal::Hold
            }
        } else {
            Signal::Hold
        }
    }
}

#[derive(Debug, Clone)]
pub struct VolumeProfile {
    volume_by_price: std::collections::HashMap<String, Decimal>,
    price_precision: u32,
}

impl VolumeProfile {
    pub fn new(price_precision: u32) -> Self {
        Self {
            volume_by_price: std::collections::HashMap::new(),
            price_precision,
        }
    }

    pub fn add_trade(&mut self, price: Decimal, volume: Decimal) {
        let price_key = format!("{:.1$}", price, self.price_precision as usize);
        *self.volume_by_price.entry(price_key).or_insert(Decimal::ZERO) += volume;
    }

    pub fn get_high_volume_nodes(&self, min_volume: Decimal) -> Vec<(Decimal, Decimal)> {
        self.volume_by_price
            .iter()
            .filter(|(_, &volume)| volume >= min_volume)
            .map(|(price_str, &volume)| {
                let price = price_str.parse::<Decimal>().unwrap_or(Decimal::ZERO);
                (price, volume)
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct ATR {
    period: usize,
    tr_values: VecDeque<Decimal>,
    atr_value: Option<Decimal>,
}

impl ATR {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            tr_values: VecDeque::new(),
            atr_value: None,
        }
    }

    pub fn update(&mut self, high: Decimal, low: Decimal, close: Decimal) -> Option<Decimal> {
        let _ = close;

        let tr = if let Some(last_close) = self.tr_values.back() {
            Decimal::max(high - low, Decimal::max((high - *last_close).abs(), (low - *last_close).abs()))
        } else {
            high - low
        };

        self.tr_values.push_back(tr);
        if self.tr_values.len() > self.period {
            self.tr_values.pop_front();
        }

        if self.tr_values.len() == self.period {
            let atr = self.tr_values.iter().sum::<Decimal>() / Decimal::from(self.period);
            self.atr_value = Some(atr);
            return Some(atr);
        }

        None
    }

    pub fn current(&self) -> Option<Decimal> {
        self.atr_value
    }
}


// Scalping-specific indicators
#[derive(Debug, Clone)]
pub struct ScalpingSignals {
    pub momentum_score: Decimal,
    pub volatility_score: Decimal,
    pub trend_strength: Decimal,
    pub entry_confidence: Decimal,
}

pub fn calculate_spread_percentage(bid: Decimal, ask: Decimal) -> Decimal {
    if bid > Decimal::ZERO {
        ((ask - bid) / bid) * Decimal::from(100)
    } else {
        Decimal::ZERO
    }
}

pub fn calculate_price_momentum(prices: &[Decimal], lookback: usize) -> Decimal {
    if prices.len() < lookback + 1 {
        return Decimal::ZERO;
    }
    
    let current = prices[prices.len() - 1];
    let previous = prices[prices.len() - 1 - lookback];
    
    if previous > Decimal::ZERO {
        ((current - previous) / previous) * Decimal::from(100)
    } else {
        Decimal::ZERO
    }
}

pub fn calculate_volatility(prices: &[Decimal], period: usize) -> Decimal {
    if prices.len() < period {
        return Decimal::ZERO;
    }
    
    let recent_prices = &prices[prices.len() - period..];
    let mean = recent_prices.iter().sum::<Decimal>() / Decimal::from(period);
    
    let variance = recent_prices
        .iter()
        .map(|price| (*price - mean) * (*price - mean))
        .sum::<Decimal>() / Decimal::from(period);
    
    decimal_sqrt(variance)
}

pub fn calculate_atr(highs: &[Decimal], lows: &[Decimal], closes: &[Decimal], period: usize) -> Decimal {
    if highs.len() < period || lows.len() < period || closes.len() < period {
        return Decimal::ZERO;
    }

    let mut atr = ATR::new(period);

    for i in 0..period {
        let high = highs[highs.len() - 1 - i];
        let low = lows[lows.len() - 1 - i];
        let close = closes[closes.len() - 1 - i];

        if let Some(_value) = atr.update(high, low, close) {
            // ATR value updated

        }
    }

    atr.current().unwrap_or(Decimal::ZERO)
}