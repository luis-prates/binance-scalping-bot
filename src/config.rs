use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub binance: BinanceConfig,
    pub trading: TradingConfig,
    pub risk_management: RiskManagementConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BinanceConfig {
    pub api_key: String,
    pub secret_key: String,
    pub testnet: bool,
    pub base_url: String,
    pub ws_url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TradingConfig {
    pub symbol: String,
    pub timeframe: String,
    pub position_size: Decimal,
    pub max_positions: u32,
    pub scalp_target_pct: Decimal, // Target profit percentage for scalp
    pub stop_loss_pct: Decimal,    // Stop loss percentage
    pub dynamic_targets: bool,     // Use dynamic targets based on volatility
    pub min_volume: Decimal,       // Minimum 24h volume to trade
    pub spread_threshold: Decimal, // Max spread to enter trade
    pub cooldown_period: u64,      // Cooldown period in seconds between trades
    pub ml_enabled: bool,
    pub ml_model_path: String,
    pub ml_prediction_threshold: f64,
    pub ml_training_interval: u64, // in seconds
    pub max_consecutive_losses: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RiskManagementConfig {
    pub max_daily_loss: Decimal,
    pub max_drawdown: Decimal,
    pub position_size_pct: Decimal, // Percentage of portfolio per trade
}

impl Default for Config {
    fn default() -> Self {
        Self {
            binance: BinanceConfig {
                api_key: String::new(),
                secret_key: String::new(),
                testnet: true,
                base_url: "https://testnet.binance.vision".to_string(),
                ws_url: "wss://testnet.binance.vision/ws".to_string(),
            },
            trading: TradingConfig {
                symbol: "BTCUSDT".to_string(),
                timeframe: "1m".to_string(),
                position_size: Decimal::from(10), // $10 per trade
                max_positions: 3,
                scalp_target_pct: Decimal::from_str_exact("0.002").unwrap(), // 0.2%
                stop_loss_pct: Decimal::from_str_exact("0.001").unwrap(),    // 0.1%
                dynamic_targets: false, // Use dynamic targets based on volatility
                min_volume: Decimal::from(1000000), // $1M daily volume
                spread_threshold: Decimal::from_str_exact("0.0005").unwrap(), // 0.05%
                cooldown_period: 60,    // 60 seconds cooldown between trades
                ml_enabled: true,
                ml_model_path: "models/trading_model.json".to_string(),
                ml_prediction_threshold: 0.6,
                ml_training_interval: 3600, // Retrain every hour
                max_consecutive_losses: 5,
            },
            risk_management: RiskManagementConfig {
                max_daily_loss: Decimal::from(100),
                max_drawdown: Decimal::from_str_exact("0.05").unwrap(), // 5%
                position_size_pct: Decimal::from_str_exact("0.01").unwrap(), // 1%
            },
        }
    }
}

impl Config {
    pub fn load() -> anyhow::Result<Self> {
        let config_str = std::fs::read_to_string("config.toml").unwrap_or_else(|_| {
            log::warn!("Config file not found, using default configuration");
            String::new()
        });

        if config_str.is_empty() {
            let default_config = Self::default();
            let toml_str = toml::to_string_pretty(&default_config)?;
            std::fs::write("config.toml", toml_str)?;
            Ok(default_config)
        } else {
            Ok(toml::from_str(&config_str)?)
        }
    }
}
