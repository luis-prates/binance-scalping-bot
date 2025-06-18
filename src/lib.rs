//! Binance Scalping Bot Library
//!
//! A high-performance cryptocurrency scalping bot for Binance
//! with comprehensive backtesting capabilities.

pub mod backtester;
pub mod binance;
pub mod config;
pub mod indicators;
pub mod ml_model;
pub mod strategy;

// Re-export commonly used types
pub use backtester::{BacktestConfig, BacktestResults, Backtester};
pub use config::Config;
pub use indicators::Signal;
pub use strategy::{Position, ScalpingStrategy};
