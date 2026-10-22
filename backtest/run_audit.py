# backtest/run_audit.py
import os
import sys
import csv
import datetime

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.backtest_engine import run_6month_simulation

def main():
    print("="*80)
    print("  STARTING COMPREHENSIVE 6-MONTH QUANT BACKTEST & INDIAN TAX AUDIT  ")
    print("="*80)

    results = run_6month_simulation(
        base_capital=10000.0,
        use_mis_leverage=True,
        leverage_multiplier=5.0
    )

    trades = results["executed_trades"]
    
    # Save trades to CSV
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "backtest_trades_6m.csv")
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Symbol", "Side", "Entry_Time", "Entry_Price", "Exit_Time", "Exit_Price",
            "Qty", "Strategy", "Grade", "Exit_Reason", "Gross_PnL", "Taxes_Paid", "Net_PnL", "Return_On_Trade_Pct"
        ])
        for t in trades:
            ret_pct = round((t.net_pnl / t.capital_used) * 100.0, 2) if t.capital_used > 0 else 0.0
            writer.writerow([
                t.symbol, t.side, t.entry_time, t.entry_price, t.exit_time, t.exit_price,
                t.qty, t.strategy, t.grade, t.exit_reason, t.gross_pnl, t.taxes, t.net_pnl, ret_pct
            ])

    print(f"\n[CSV SAVED] Detailed audit log of all {len(trades)} trades exported to:")
    print(f"  --> {csv_path}\n")

    # Monthly Breakdown
    monthly_data = {}
    for t in trades:
        m_key = t.entry_time[:7] # YYYY-MM
        m_stats = monthly_data.setdefault(m_key, {"trades": 0, "wins": 0, "losses": 0, "gross": 0.0, "taxes": 0.0, "net": 0.0})
        m_stats["trades"] += 1
        if t.net_pnl > 0:
            m_stats["wins"] += 1
        else:
            m_stats["losses"] += 1
        m_stats["gross"] += t.gross_pnl
        m_stats["taxes"] += t.taxes
        m_stats["net"] += t.net_pnl

    print("="*80)
    print("                     MONTH-BY-MONTH PERFORMANCE AUDIT                        ")
    print("="*80)
    print(f"{'Month':<10} | {'Trades':<7} | {'Win Rate':<9} | {'Gross PnL':<12} | {'Taxes Paid':<11} | {'Net Take-Home':<14} | {'Net Return':<10}")
    print("-"*80)
    for m in sorted(monthly_data.keys()):
        s = monthly_data[m]
        wr = (s["wins"] / s["trades"] * 100.0) if s["trades"] > 0 else 0.0
        ret = (s["net"] / 10000.0) * 100.0
        print(f"{m:<10} | {s['trades']:<7} | {wr:>7.1f}% | Rs. {s['gross']:>9.2f} | Rs. {s['taxes']:>8.2f} | Rs. {s['net']:>11.2f} | {ret:>+8.1f}%")

    print("="*80)
    print("                 6-MONTH MASTER QUANTITATIVE SUMMARY                        ")
    print("="*80)
    print(f"• Initial Cash Capital:       Rs. {results['base_capital']:,.2f}")
    print(f"• Final Account Balance:      Rs. {results['final_capital']:,.2f}")
    print(f"• Total Net Take-Home PnL:    Rs. {results['total_net_pnl']:+,.2f}  (Return on Capital: {results['return_on_capital_pct']:+.1f}%)")
    print(f"• Total Trades Executed:      {results['total_trades']}")
    print(f"• Win Rate:                   {results['win_rate_pct']:.2f}%  ({results['winning_trades']} Wins / {results['losing_trades']} Losses)")
    print(f"• Breakeven Shield Saves:     {results['breakeven_trades']} trades exited at Rs. 0 net loss via Never-Red Breakeven")
    print(f"• Profit Factor:              {results['profit_factor']:.2f} (Gross Win / Gross Loss)")
    print(f"• Payoff Ratio (Avg W / L):   {results['payoff_ratio']:.2f} (Avg Win: Rs. {results['avg_win']:,.2f} | Avg Loss: Rs. {results['avg_loss']:,.2f})")
    print(f"• Expectancy per Trade:       Rs. {results['expectancy_per_trade']:+,.2f}")
    print(f"• Max Drawdown:               Rs. {results['max_drawdown_rupees']:,.2f} ({results['max_drawdown_pct']:.2f}%)")
    print(f"• Total Statutory Taxes Paid: Rs. {results['total_taxes_paid']:,.2f} (Brokerage + STT + NSE + SEBI + Stamp + GST)")
    print("="*80)

if __name__ == "__main__":
    main()
