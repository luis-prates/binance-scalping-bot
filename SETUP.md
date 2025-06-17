# Quick Start Guide

## Prerequisites

1. **Rust Installation**: Make sure you have Rust installed from [rustup.rs](https://rustup.rs/)
2. **Binance Account**: Create a Binance account if you don't have one
3. **API Keys**: Generate API keys with trading permissions

## Step-by-Step Setup

### 1. Initial Build
```bash
# Build the project
cargo build --release

# Run once to generate config file
cargo run
```

### 2. Configure API Keys
Edit the generated `config.toml` file:

```toml
[binance]
api_key = "YOUR_ACTUAL_API_KEY_HERE"
secret_key = "YOUR_ACTUAL_SECRET_KEY_HERE"
testnet = true  # KEEP THIS TRUE FOR TESTING!
base_url = "https://testnet.binance.vision"
ws_url = "wss://testnet.binance.vision/ws"

[trading]
symbol = "BTCUSDT"
timeframe = "1m"
position_size = "10"          # Start with $10 per trade
max_positions = 3
scalp_target_pct = "0.002"    # 0.2% profit target
stop_loss_pct = "0.001"       # 0.1% stop loss  
min_volume = "1000000"        # Minimum 24h volume
spread_threshold = "0.0005"   # Max 0.05% spread

[risk_management]
max_daily_loss = "100"
max_drawdown = "0.05"         # 5% max drawdown
position_size_pct = "0.01"    # 1% of portfolio per trade
max_consecutive_losses = 5
```

### 3. Get Testnet API Keys

1. Go to [Binance Testnet](https://testnet.binance.vision/)
2. Log in with your GitHub account
3. Create API keys with trading permissions
4. Add them to your config.toml

### 4. Run the Bot

```bash
# Run with info-level logging
cargo run --release

# Run with debug logging for more details
RUST_LOG=debug cargo run --release
```

## Safety First! ⚠️

### IMPORTANT: Start with Testnet
- **ALWAYS** test on Binance Testnet first
- Keep `testnet = true` until you're confident
- Start with very small position sizes
- Monitor the bot closely

### Live Trading Transition
Only switch to live trading after:
1. Extensive testnet testing
2. Understanding all parameters
3. Starting with minimal position sizes
4. Having stop-loss mechanisms in place

```toml
# For live trading (use with extreme caution)
[binance]
api_key = "your_live_api_key"
secret_key = "your_live_secret_key"
testnet = false
base_url = "https://api.binance.com"
ws_url = "wss://stream.binance.com:9443/ws"
```

## Bot Behavior

### Entry Signals
The bot enters trades when at least 2 of these conditions are met:
- EMA crossover (5 > 13 for buy, 5 < 13 for sell)
- RSI extremes (< 30 for buy, > 70 for sell)
- MACD momentum (positive histogram for buy, negative for sell)
- Bollinger Bands (price at lower band for buy, upper for sell)
- Volume confirmation (20% above average)
- Price momentum (> 0.1% for buy, < -0.1% for sell)

### Exit Strategy
- **Profit Target**: Closes at 0.2% profit by default
- **Stop Loss**: Closes at 0.1% loss by default
- **Risk Management**: Stops trading after 5 consecutive losses

### Monitoring
The bot logs:
- Trade entries and exits
- Performance statistics every 10 trades
- Warnings for risky market conditions
- Current P&L and win rate

## Troubleshooting

### Common Issues

1. **"API key not found"**
   - Check your config.toml has correct API keys
   - Ensure API keys have trading permissions

2. **"Insufficient balance"**
   - Check you have enough USDT in your account
   - Reduce position_size in config

3. **"Symbol not found"**
   - Verify the trading symbol exists on Binance
   - Check symbol format (e.g., "BTCUSDT" not "BTC/USDT")

4. **High spread warnings**
   - Normal during low liquidity periods
   - Consider increasing spread_threshold

### Getting Help

1. Check the logs for specific error messages
2. Verify your configuration matches the examples
3. Test with minimal position sizes first
4. Use debug logging: `RUST_LOG=debug cargo run`

## Disclaimer

**⚠️ FINANCIAL RISK WARNING ⚠️**

This bot is for educational purposes. Cryptocurrency trading involves substantial risk of loss. The authors are not responsible for any financial losses. Key risks:

- **Market Risk**: Crypto markets are highly volatile
- **Technical Risk**: Bugs or outages can cause losses  
- **Strategy Risk**: Past performance doesn't guarantee future results
- **Execution Risk**: Network issues can prevent proper trade execution

**Never trade with money you can't afford to lose!**

Start small, learn the system, and gradually increase position sizes only after proving profitability over extended periods.

