use anyhow::Result;
use binance_scalping_bot::backtester::{BacktestConfig, Backtester};
use binance_scalping_bot::binance::BinanceClient;
use binance_scalping_bot::config::Config;
use chrono::Utc;
use clap::{Arg, Command};
use log::info;
use rust_decimal::Decimal;
use std::str::FromStr;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    // Parse command line arguments
    let matches = Command::new("Binance Scalping Bot Backtester")
        .version("1.0")
        .about("Backtest the scalping strategy against historical data")
        .arg(
            Arg::new("start-date")
                .long("start-date")
                .help("Start date for backtest (YYYY-MM-DD)")
                .default_value("2025-06-05")
                .required(false),
        )
        .arg(
            Arg::new("end-date")
                .long("end-date")
                .help("End date for backtest (YYYY-MM-DD)")
                .default_value("2025-06-15")
                .required(false),
        )
        .arg(
            Arg::new("initial-balance")
                .long("initial-balance")
                .help("Initial balance for backtest")
                .default_value("1000")
                .required(false),
        )
        .arg(
            Arg::new("symbol")
                .long("symbol")
                .help("Trading symbol (e.g., BTCUSDT)")
                .default_value("BTCUSDT")
                .required(false),
        )
        .arg(
            Arg::new("interval")
                .long("interval")
                .help("Data interval (1m, 5m, 15m, 1h)")
                .default_value("1m")
                .required(false),
        )
        .arg(
            Arg::new("commission")
                .long("commission")
                .help("Commission rate (e.g., 0.001 for 0.1%)")
                .default_value("0.001")
                .required(false),
        )
        .arg(
            Arg::new("slippage")
                .long("slippage")
                .help("Slippage rate (e.g., 0.0001 for 0.01%)")
                .default_value("0.0001")
                .required(false),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose logging")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("export")
                .long("export")
                .help("Export results to CSV file")
                .value_name("FILE")
                .required(false),
        )
        .arg(
            Arg::new("filter-interval")
                .long("filter-interval")
                .help("Filter data interval (1m, 5m, 15m, 1h)")
                .default_value("5m")
                .required(false),
        )
        .get_matches();

    // Set up verbose logging if requested
    if matches.get_flag("verbose") {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Debug)
            .init();
    }

    // Load configuration
    let mut config = Config::load()?;

    // Override symbol if provided
    if let Some(symbol) = matches.get_one::<String>("symbol") {
        config.trading.symbol = symbol.clone();
    }

    info!("Starting backtest for {}", config.trading.symbol);

    // Create backtest configuration
    let backtest_config = BacktestConfig {
        start_date: matches.get_one::<String>("start-date").unwrap().clone(),
        end_date: matches.get_one::<String>("end-date").unwrap().clone(),
        initial_balance: Decimal::from_str(matches.get_one::<String>("initial-balance").unwrap())?,
        commission_rate: Decimal::from_str(matches.get_one::<String>("commission").unwrap())?,
        slippage: Decimal::from_str(matches.get_one::<String>("slippage").unwrap())?,
        data_interval: matches.get_one::<String>("interval").unwrap().clone(),
        max_klines_per_request: 1000,
        filter_data_interval: matches
            .get_one::<String>("filter-interval")
            .unwrap()
            .clone(),
    };

    // Create Binance client
    let client = BinanceClient::new(
        config.binance.api_key.clone(),
        config.binance.secret_key.clone(),
        config.binance.base_url.clone(),
    );

    // Create and run backtester
    let mut backtester = Backtester::new(client, backtest_config.clone(), config.trading.clone());

    info!(
        "Running backtest from {} to {}",
        backtest_config.start_date, backtest_config.end_date
    );
    let results = backtester.run_backtest().await?;

    // Display results
    print_results(&results);

    // Export results if requested
    if let Some(export_file) = matches.get_one::<String>("export") {
        export_results_to_csv(&results, export_file)?;
        info!("Results exported to {}", export_file);
    }

    Ok(())
}

fn print_results(results: &binance_scalping_bot::backtester::BacktestResults) {
    println!();
    println!("==================== BACKTEST RESULTS ====================");
    println!();

    // Performance Summary
    println!("📊 PERFORMANCE SUMMARY");
    println!("   Initial Balance:    ${:.2}", results.initial_balance);
    println!("   Final Balance:      ${:.2}", results.final_balance);
    println!("   Total Return:       ${:.2}", results.total_return);
    println!("   Return Percentage:  {:.2}%", results.total_return_pct);
    println!("   Total Commission:   ${:.2}", results.total_commission);
    println!();

    // Trading Statistics
    println!("📈 TRADING STATISTICS");
    println!("   Total Trades:       {}", results.total_trades);
    println!("   Winning Trades:     {}", results.winning_trades);
    println!("   Losing Trades:      {}", results.losing_trades);
    println!("   Win Rate:           {:.2}%", results.win_rate);
    println!("   Average Win:        ${:.4}", results.avg_win);
    println!("   Average Loss:       ${:.4}", results.avg_loss);
    println!("   Largest Win:        ${:.4}", results.largest_win);
    println!("   Largest Loss:       ${:.4}", results.largest_loss);
    println!("   Profit Factor:      {:.2}", results.profit_factor);
    println!();

    // Risk Metrics
    println!("⚠️  RISK METRICS");
    println!("   Max Drawdown:       ${:.2}", results.max_drawdown);
    println!("   Max Drawdown %:     {:.2}%", results.max_drawdown_pct);
    println!("   Sharpe Ratio:       {:.2}", results.sharpe_ratio);
    println!();

    // Recent Trades (last 10)
    println!("🔄 RECENT TRADES (Last 10)");
    let recent_trades = results.trades.iter().rev().take(10).collect::<Vec<_>>();

    if !recent_trades.is_empty() {
        println!("   ID    Side  Entry Price  Exit Price   P&L      Reason");
        println!("   ----------------------------------------------------------------");

        let open_text = "Open".to_string();

        for trade in recent_trades {
            let exit_price = trade
                .exit_price
                .map(|p| format!("{:.4}", p))
                .unwrap_or_else(|| "N/A".to_string());

            let pnl = trade
                .pnl
                .map(|p| format!("{:+.4}", p))
                .unwrap_or_else(|| "N/A".to_string());

            let reason = trade.exit_reason.as_ref().unwrap_or(&open_text);

            println!(
                "   {:3}   {:4}  {:10.4}  {:>10}  {:>8}  {}",
                trade.id, trade.side, trade.entry_price, exit_price, pnl, reason
            );
        }
    } else {
        println!("   No trades executed during backtest period.");
    }

    println!();
    println!("==========================================================");

    // Performance Analysis
    if results.total_return_pct > Decimal::ZERO {
        println!("✅ Strategy was PROFITABLE during the backtest period");
    } else {
        println!("❌ Strategy was UNPROFITABLE during the backtest period");
    }

    if results.win_rate > 50.0 {
        println!("✅ Win rate above 50%");
    } else {
        println!("⚠️  Win rate below 50%");
    }

    if results.profit_factor > Decimal::ONE {
        println!("✅ Positive profit factor (wins > losses)");
    } else {
        println!("❌ Negative profit factor (losses > wins)");
    }

    if results.max_drawdown_pct < Decimal::from(10) {
        println!("✅ Acceptable drawdown (< 10%)");
    } else {
        println!("⚠️  High drawdown (> 10%)");
    }

    println!();
}

fn export_results_to_csv(
    results: &binance_scalping_bot::backtester::BacktestResults,
    filename: &str,
) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(filename)?;

    // Write CSV header
    writeln!(
        file,
        "trade_id,symbol,side,entry_time,exit_time,entry_price,exit_price,quantity,commission,pnl,exit_reason"
    )?;

    // Write trade data
    for trade in &results.trades {
        let exit_time = trade
            .exit_time
            .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| "N/A".to_string());

        let exit_price = trade
            .exit_price
            .map(|p| p.to_string())
            .unwrap_or_else(|| "N/A".to_string());

        let pnl = trade
            .pnl
            .map(|p| p.to_string())
            .unwrap_or_else(|| "N/A".to_string());

        let open_text = "Open".to_string();

        let exit_reason = trade.exit_reason.as_ref().unwrap_or(&open_text);

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{}",
            trade.id,
            trade.symbol,
            trade.side,
            trade.entry_time.format("%Y-%m-%d %H:%M:%S"),
            exit_time,
            trade.entry_price,
            exit_price,
            trade.quantity,
            trade.commission,
            pnl,
            exit_reason
        )?;
    }

    Ok(())
}
