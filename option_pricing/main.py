import argparse
import sys
import os

# Local imports
from config import CONFIG
from data_handler import get_data
from strategies import sample_strategy
from backtester import Backtester
from metrics import calculate_metrics
from hedging import run_hedging_strategy

def run_backtest():
    print("\n=== BACKTEST MODE ===")
    print(f"Fetching data for {CONFIG['ticker']} from {CONFIG['start_date']} to {CONFIG['end_date']}...")
    
    # 1. Get data
    df = get_data(CONFIG['ticker'], CONFIG['start_date'], CONFIG['end_date'], CONFIG['interval'])
    if df.empty:
        print("❌ No data returned. Check ticker or date range.")
        sys.exit(1)
    
    print("✅ Data fetched successfully!")
    
    # 2. Run strategy
    print(f"Running strategy: {CONFIG['strategy_name']}...")
    signals = sample_strategy(df)  # Replace with actual strategy from strategies.py
    
    # 3. Backtest
    bt = Backtester(initial_capital=CONFIG['initial_capital'])
    results = bt.run(df, signals)
    
    # 4. Metrics
    print("\n=== PERFORMANCE METRICS ===")
    metrics = calculate_metrics(results["equity_curve"])
    for k, v in metrics.items():
        print(f"{k}: {v}")
    
    # 5. Plot equity curve
    results["equity_curve"].plot(title=f"Equity Curve - {CONFIG['strategy_name']}")
    
    print("\n✅ Backtest completed.")

def run_hedge():
    print("\n=== HEDGING MODE ===")
    run_hedging_strategy(CONFIG["hedge_params"])
    print("\n✅ Hedging simulation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options Pricing & Hedging Project")
    parser.add_argument("--mode", type=str, required=True, choices=["backtest", "hedge"],
                        help="Choose mode: backtest or hedge")
    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest()
    elif args.mode == "hedge":
        run_hedge()
