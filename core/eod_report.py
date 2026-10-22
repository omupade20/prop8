# core/eod_report.py
"""
JARVIS Institutional End-of-Day (EOD) HTML Visual Trade Report Generator.
Generates an executive, dark-mode performance report card at 15:15 IST.
Highlights:
- Gross P&L vs Indian Taxes vs Net Take-Home In-Pocket P&L
- Win-Rate & Risk Metrics
- Individual Trade Breakdown Table
"""

import os
import csv
import datetime
import webbrowser
from typing import List, Dict, Any


def generate_eod_html_report(csv_path: str = "logs/trades/all_trades.csv", auto_open: bool = False) -> str:
    """
    Parses today's trades from the master CSV and produces a standalone HTML visual report card.
    """
    today_str = datetime.date.today().isoformat()
    trades: List[Dict[str, str]] = []

    if os.path.exists(csv_path):
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("date") == today_str:
                    trades.append(row)

    total_trades = len(trades)
    gross_pnl = 0.0
    total_charges = 0.0
    net_pnl = 0.0
    wins = 0
    losses = 0

    for t in trades:
        try:
            g = float(t.get("gross_pnl", 0.0))
            c = float(t.get("charges", 0.0))
            n = float(t.get("net_pnl", 0.0))
            gross_pnl += g
            total_charges += c
            net_pnl += n
            if n > 0:
                wins += 1
            else:
                losses += 1
        except Exception:
            pass

    win_rate = (wins / max(1, total_trades)) * 100.0
    status_color = "#10b981" if net_pnl >= 0 else "#ef4444"
    status_badge = "NET GREEN SESSION 🟢" if net_pnl >= 0 else "NET RED SESSION 🔴"

    rows_html = ""
    for idx, t in enumerate(trades, 1):
        try:
            t_net = float(t.get("net_pnl", 0.0))
            row_color = "#10b981" if t_net >= 0 else "#ef4444"
            sym = t.get("instrument", "").split("|")[-1].replace("NSE_EQ", "").strip(":")
            rows_html += f"""
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding: 12px; color: #94a3b8;">#{idx}</td>
                <td style="padding: 12px; font-weight: bold; color: #f8fafc;">{sym}</td>
                <td style="padding: 12px; color: {'#10b981' if t.get('side') == 'BUY' else '#ef4444'}; font-weight: bold;">{t.get('side')}</td>
                <td style="padding: 12px; color: #cbd5e1;">{t.get('quantity')}</td>
                <td style="padding: 12px; color: #cbd5e1;">Rs. {float(t.get('entry_price', 0)):.2f}</td>
                <td style="padding: 12px; color: #cbd5e1;">Rs. {float(t.get('exit_price', 0)):.2f}</td>
                <td style="padding: 12px; color: #94a3b8;">{t.get('entry_time')} - {t.get('exit_time')}</td>
                <td style="padding: 12px; color: #cbd5e1;">Rs. {float(t.get('gross_pnl', 0)):+,.2f}</td>
                <td style="padding: 12px; color: #f59e0b;">Rs. {float(t.get('charges', 0)):,.2f}</td>
                <td style="padding: 12px; font-weight: bold; color: {row_color};">Rs. {t_net:+,.2f}</td>
                <td style="padding: 12px; color: #38bdf8; font-size: 13px;">{t.get('exit_reason')}</td>
            </tr>
            """
        except Exception:
            pass

    if not trades:
        rows_html = """
        <tr>
            <td colspan="11" style="padding: 30px; text-align: center; color: #64748b;">
                No trades were executed today or market conditions did not meet the strict 70% AI Probability hurdle.
            </td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JARVIS Quant Report - {today_str}</title>
    <style>
        body {{
            background: #090d16;
            color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 32px 16px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 24px;
            margin-bottom: 32px;
        }}
        .badge {{
            background: {status_color}22;
            color: {status_color};
            border: 1px solid {status_color};
            padding: 6px 16px;
            border-radius: 9999px;
            font-weight: bold;
            font-size: 14px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }}
        .card {{
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 20px;
        }}
        .card-label {{
            color: #94a3b8;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }}
        .card-value {{
            font-size: 26px;
            font-weight: 800;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            background: #111827;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #1f2937;
        }}
        th {{
            background: #1a2234;
            padding: 14px 12px;
            font-size: 13px;
            color: #94a3b8;
            text-transform: uppercase;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 style="margin: 0 0 6px 0; font-size: 28px;">⚡ JARVIS QUANT EOD REPORT</h1>
                <div style="color: #64748b; font-size: 14px;">Date: {today_str} | Market: NSE Equity Intraday Cash (MIS)</div>
            </div>
            <div class="badge">{status_badge}</div>
        </div>

        <div class="kpi-grid">
            <div class="card">
                <div class="card-label">Net P&L (In Account)</div>
                <div class="card-value" style="color: {status_color};">Rs. {net_pnl:+,.2f}</div>
            </div>
            <div class="card">
                <div class="card-label">Gross P&L</div>
                <div class="card-value" style="color: #cbd5e1;">Rs. {gross_pnl:+,.2f}</div>
            </div>
            <div class="card">
                <div class="card-label">Indian Taxes & Brokerage</div>
                <div class="card-value" style="color: #f59e0b;">-Rs. {total_charges:,.2f}</div>
            </div>
            <div class="card">
                <div class="card-label">Win Rate & Trades</div>
                <div class="card-value" style="color: #38bdf8;">{win_rate:.0f}% <span style="font-size: 16px; color: #94a3b8;">({wins}W / {losses}L / {total_trades}T)</span></div>
            </div>
        </div>

        <h3 style="margin: 0 0 16px 0; font-size: 18px; color: #e2e8f0;">Session Trade Executions</h3>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Asset</th>
                    <th>Side</th>
                    <th>Qty</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>Time</th>
                    <th>Gross</th>
                    <th>Taxes</th>
                    <th>Net P&L</th>
                    <th>Exit Reason</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

        <div style="margin-top: 32px; text-align: center; color: #475569; font-size: 13px;">
            Institutional Ultron 5-AI Engine Architecture | Verified Indian MIS Cash Protocol
        </div>
    </div>
</body>
</html>
"""
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_file = os.path.join(reports_dir, f"eod_report_{today_str}.html")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[EOD REPORT] Saved visual performance card: {report_file}")
    if auto_open:
        try:
            webbrowser.open(f"file://{os.path.abspath(report_file)}")
        except Exception:
            pass
    return report_file
