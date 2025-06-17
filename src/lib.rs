//! Binance Scalping Bot Library
//! 
//! A high-performance cryptocurrency scalping bot for Binance
//! with comprehensive backtesting capabilities.

pub mod binance;
pub mod config;
pub mod indicators;
pub mod strategy;
pub mod backtester;

// Re-export commonly used types
pub use config::Config;
pub use strategy::{ScalpingStrategy, Position};
pub use indicators::Signal;
pub use backtester::{Backtester, BacktestConfig, BacktestResults};

