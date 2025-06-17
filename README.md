# Binance Scalping Bot

A high-performance cryptocurrency scalping bot built in Rust for Binance. This bot implements multiple technical analysis indicators and risk management strategies to execute profitable short-term trades.

## Features

### Trading Strategy
- **Multi-indicator Analysis**: Combines EMA crossovers, RSI, MACD, and Bollinger Bands
- **Scalping Focus**: Optimized for quick entries and exits (0.1-0.5% profit targets)
- **Volume Confirmation**: Validates signals with volume analysis
- **Momentum Detection**: Uses price momentum to improve entry timing

### Risk Management
- **Position Sizing**: Configurable position sizes based on account balance
- **Stop Losses**: Automatic stop-loss orders to limit downside
- **Maximum Positions**: Limits concurrent positions to manage risk
- **Consecutive Loss Protection**: Stops trading after too many consecutive losses
- **Spread Filtering**: Avoids trading when spreads are too wide

### Technical Features
- **Real-time Data**: Uses Binance WebSocket and REST APIs
- **Async Architecture**: Built with Tokio for high performance
- **Comprehensive Logging**: Detailed logs for monitoring and debugging
- **Graceful Shutdown**: Handles Ctrl+C and cleanup properly
- **Configuration Management**: Easy-to-modify TOML configuration

## Prerequisites

1. **Rust Installation**: Install Rust from [rustup.rs](https://rustup.rs/)
2. **Binance Account**: Create a Binance account and enable API access
3. **API Keys**: Generate API keys with trading permissions

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd binance-scalping-bot
```

2. Build the project:
```bash
cargo build --release
```

## Configuration

1. Run the bot once to generate the default configuration:
```bash
cargo run
```

2. Edit the generated `config.toml` file:
```toml
[binance]
api_key = "your_api_key_here"
secret_key = "your_secret_key_here"
testnet = true  # Set to false for live trading
base_url = "https://testnet.binance.vision"  # Use https://api.binance.com for live
ws_url = "wss://testnet.binance.vision/ws"

[trading]
symbol = "BTCUSDT"
timeframe = "1m"
position_size = 10.0  # USD amount per trade
max_positions = 3
scalp_target_pct = 0.002  # 0.2% profit target
stop_loss_pct = 0.001     # 0.1% stop loss
min_volume = 1000000.0    # Minimum 24h volume
spread_threshold = 0.0005 # Maximum spread (0.05%)

[risk_management]
max_daily_loss = 100.0
max_drawdown = 0.05
position_size_pct = 0.01
max_consecutive_losses = 5
```

## Usage

### Testnet Trading (Recommended for testing)
1. Create a Binance Testnet account at [testnet.binance.vision](https://testnet.binance.vision/)
2. Generate testnet API keys
3. Configure the bot with testnet settings
4. Run the bot:
```bash
cargo run --release
```

### Live Trading (Use with caution)
1. Set `testnet = false` in config.toml
2. Update the base_url to the live Binance API
3. Use your live API keys
4. Start with small position sizes

## Strategy Details

### Entry Signals
The bot generates buy/sell signals when at least 2 of the following conditions are met:

1. **EMA Crossover**: Fast EMA (5) above/below Slow EMA (13)
2. **RSI Extremes**: RSI below 30 (oversold) or above 70 (overbought)
3. **MACD Momentum**: MACD line above/below signal line with positive/negative histogram
4. **Bollinger Bands**: Price touching lower/upper bands
5. **Volume Confirmation**: Current volume 20% above average
6. **Price Momentum**: 3-period momentum above/below threshold

### Exit Strategy
- **Profit Target**: Close position when profit target is reached (default 0.2%)
- **Stop Loss**: Close position when stop loss is hit (default 0.1%)
- **Time-based**: Optional time-based exits for stale positions

### Risk Controls
- Maximum 3 concurrent positions
- Stop trading after 5 consecutive losses
- Avoid trading when spread > 0.05%
- Require minimum 24h volume
- Position sizing based on account balance

## Performance Monitoring

The bot provides real-time statistics:
- Total trades executed
- Win rate percentage
- Total P&L
- Current open positions

## Important Warnings

⚠️ **CRYPTOCURRENCY TRADING RISKS**:
- Trading cryptocurrencies involves substantial risk of loss
- Past performance does not guarantee future results
- Never trade with money you can't afford to lose
- Start with small amounts and paper trading

⚠️ **BOT-SPECIFIC RISKS**:
- Algorithmic trading can lead to rapid losses
- Market conditions can change quickly
- Technical indicators can give false signals
- Always monitor the bot's performance

⚠️ **RECOMMENDATIONS**:
- Always test on testnet first
- Start with very small position sizes
- Monitor the bot closely, especially initially
- Have manual override capabilities
- Regularly review and adjust parameters

## Customization

### Adding New Indicators
1. Implement the indicator in `src/indicators.rs`
2. Add it to the strategy in `src/strategy.rs`
3. Update the signal generation logic

### Modifying Strategy
- Adjust indicator parameters in the strategy constructor
- Modify signal requirements in `analyze_market()`
- Change position sizing logic in `calculate_position_size()`

### Risk Management
- Update risk parameters in the configuration
- Modify risk checks in `should_trade()`
- Adjust position limits and loss thresholds

## Troubleshooting

### Common Issues
1. **API Connection Errors**: Check API keys and network connectivity
2. **Insufficient Balance**: Ensure adequate funds for trading
3. **Symbol Not Found**: Verify the trading symbol is correct
4. **Rate Limits**: The bot includes rate limiting, but monitor for 429 errors

### Logging
Set the log level using the `RUST_LOG` environment variable:
```bash
RUST_LOG=debug cargo run
```

## Development

### Project Structure
```
src/
├── main.rs          # Main bot orchestration
├── binance.rs       # Binance API client
├── config.rs        # Configuration management
├── indicators.rs    # Technical analysis indicators
└── strategy.rs      # Trading strategy implementation
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. The authors are not responsible for any financial losses incurred through the use of this bot. Use at your own risk and always do your own research before trading.

