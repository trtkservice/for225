#!/usr/bin/env python3
"""
Generate HTML Dashboard from predictions.json
Supports A/B Testing Visualization (Antigravity vs Raptor)
"""

import json
import os
from datetime import datetime
import html as html_lib

# --- Configuration ---

class DashboardConfig:
    OUTPUT_FILE = "data/dashboard.html"
    DATA_FILE = "data/predictions.json"
    
    CSS = """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #eee; min-height: 100vh; padding: 20px; }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 20px; font-size: 2em; background: linear-gradient(90deg, #00d4ff, #7c3aed, #ff6b6b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        
        /* A/B Comparison Layout */
        .strategy-container { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 40px; }
        .strategy-col { background: rgba(255,255,255,0.02); border-radius: 20px; padding: 20px; border: 1px solid rgba(255,255,255,0.05); }
        .strategy-header { text-align: center; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); }
        .strategy-name { font-size: 1.5em; font-weight: bold; margin-bottom: 5px; }
        .strategy-desc { font-size: 0.9em; color: #888; }
        
        .col-normal { border-top: 5px solid #00d4ff; }
        .col-raptor { border-top: 5px solid #ff6b6b; }
        
        .stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 20px; }
        .stat-card { background: rgba(0,0,0,0.2); border-radius: 10px; padding: 15px; text-align: center; }
        .stat-card h3 { font-size: 0.8em; color: #aaa; margin-bottom: 5px; }
        .stat-card .value { font-size: 1.4em; font-weight: bold; }
        
        .profit { color: #00d4ff; }
        .loss { color: #ff6b6b; }
        .neutral { color: #888; }
        
        .chart-container { background: rgba(0,0,0,0.2); border-radius: 15px; padding: 20px; margin-bottom: 40px; border: 1px solid rgba(255,255,255,0.05); height: 400px; }
        
        table { width: 100%; border-collapse: collapse; font-size: 0.9em; margin-top: 10px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.05); }
        th { background: rgba(0,0,0,0.3); font-weight: 600; color: #aaa; }
        
        .signal { padding: 3px 10px; border-radius: 10px; font-weight: bold; font-size: 0.8em; }
        .signal.long { background: rgba(0, 212, 255, 0.2); color: #00d4ff; }
        .signal.short { background: rgba(255, 107, 107, 0.2); color: #ff6b6b; }
        
        .position-card { background: rgba(0,0,0,0.3); border-radius: 10px; padding: 15px; margin-bottom: 20px; border-left: 5px solid #888; }
        .position-card.long { border-left-color: #00d4ff; }
        .position-card.short { border-left-color: #ff6b6b; }
        .position-row { display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 0.9em; }
        
        .section-title { text-align: center; margin: 40px 0 20px; font-size: 1.5em; border-bottom: 2px solid rgba(255,255,255,0.1); padding-bottom: 10px; }
        
        /* Prediction Log Styles */
        .score-grid { display: grid; grid-template-columns: repeat(3, 1fr) auto; gap: 8px; font-size: 0.8em; align-items: center; }
        .score-item b { margin-left: 3px; }
        
        @media (max-width: 1000px) {
            .strategy-container { grid-template-columns: 1fr; }
        }
    """

def load_data():
    if os.path.exists(DashboardConfig.DATA_FILE):
        with open(DashboardConfig.DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"predictions": [], "portfolios": {}, "shadow_portfolio": {}}

def _generate_strategy_block(pf, name, desc, css_class):
    capital = pf.get("capital", 100000)
    initial_capital = 100000
    total_pnl = capital - initial_capital
    pnl_pct = (total_pnl / initial_capital) * 100
    
    trades = pf.get("trades", [])
    win_trades = [t for t in trades if t.get("pnl_yen", 0) > 0]
    win_rate = (len(win_trades) / len(trades) * 100) if trades else 0.0
    
    # Position
    pos = pf.get("position")
    pos_html = ""
    if pos:
        d = pos["direction"]
        cls = "long" if d == "LONG" else "short"
        pos_html = f"""
        <div class="position-card {cls}">
            <div style="font-weight:bold; margin-bottom:10px; color:#fff">{d} @ {pos['entry_price']:,}</div>
            <div class="position-row"><span>Stop:</span> <span>{pos['stop']:,}</span></div>
            <div class="position-row"><span>Target:</span> <span>{pos['target']:,}</span></div>
            <div class="position-row"><span>Date:</span> <span>{pos['entry_date']}</span></div>
        </div>
        """
    else:
        pos_html = """<div class="position-card"><div style="color:#aaa; text-align:center">No Open Position</div></div>"""
    
    # Trade Table
    trade_rows = ""
    for t in reversed(trades[-5:]):
        pnl_cls = "profit" if t['pnl_yen'] > 0 else "loss"
        reason = t.get('close_reason', 'N/A')
        trade_rows += f"""
        <tr>
            <td>{t['exit_date'][5:]}</td>
            <td><span class="signal {'long' if t['direction']=='LONG' else 'short'}">{t['direction'][0]}</span></td>
            <td class="{pnl_cls}">¥{t['pnl_yen']:+,.0f}</td>
            <td style="font-size:0.8em; color:#aaa">{reason}</td>
        </tr>
        """
        
    return f"""
    <div class="strategy-col {css_class}">
        <div class="strategy-header">
            <div class="strategy-name">{name}</div>
            <div class="strategy-desc">{desc}</div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Capital</h3>
                <div class="value">¥{capital:,.0f}</div>
            </div>
            <div class="stat-card">
                <h3>Return</h3>
                <div class="value {'profit' if pnl_pct>=0 else 'loss'}">{pnl_pct:+.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>Win Rate</h3>
                <div class="value">{win_rate:.0f}%</div>
            </div>
            <div class="stat-card">
                <h3>Trades</h3>
                <div class="value">{len(trades)}</div>
            </div>
        </div>
        
        <h3>📍 Current Position</h3>
        {pos_html}
        
        <h3>💹 Recent Trades</h3>
        <table>
            <thead><tr><th>Date</th><th>Dir</th><th>PnL</th><th>Reason</th></tr></thead>
            <tbody>{trade_rows if trade_rows else '<tr><td colspan="4" style="text-align:center">No trades yet</td></tr>'}</tbody>
        </table>
    </div>
    """, trades

def _get_css():
    return DashboardConfig.CSS

def _generate_prediction_log(predictions):
    pred_rows = ""
    for p in reversed(predictions[-15:]):
        direction = p["prediction"]["direction"]
        cls = "long" if direction == "LONG" else "short" if direction == "SHORT" else "neutral"
        
        s = p["scores"]
        score_html = f"""
        <div class="score-grid">
            <span class="score-item">T:<b class="{'profit' if s['trend_score']>0 else 'loss'}">{s['trend_score']:.1f}</b></span>
            <span class="score-item">M:<b class="{'profit' if s['momentum_score']>0 else 'loss'}">{s['momentum_score']:.1f}</b></span>
            <span class="score-item">V:<b class="{'profit' if s['volatility_score']>0 else 'loss'}">{s['volatility_score']:.1f}</b></span>
            <span>Total:<b class="{'profit' if s['total_score']>0 else 'loss'}">{s['total_score']:.2f}</b></span>
        </div>
        """
        
        pred_rows += f"""
        <tr>
            <td>{p['timestamp']}</td>
            <td><span class="signal {cls}">{direction}</span></td>
            <td>{p['atr']}</td>
            <td>{score_html}</td>
            <td style="font-size:0.85em; color:#ccc">{p['prediction']['reasoning']}</td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Antigravity A/B Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>{DashboardConfig.CSS}</style>
</head>
<body>
    <div class="container">
        <h1>🚀 Antigravity Battle Arena</h1>
        <p style="text-align:center; color:#888; margin-bottom:30px">A/B Testing: Optimized (Normal) vs Raptor (Wide)</p>
        
        <div class="chart-container">
            <canvas id="performanceChart"></canvas>
        </div>
        
        <div class="strategy-container">
            {html_opt}
            {html_rap}
        </div>
        
        <h2 class="section-title">🔮 AI Market Analysis Log</h2>
        <table>
            <thead><tr><th>Time</th><th>Signal</th><th>ATR</th><th>Scores (T/M/V)</th><th>Reasoning</th></tr></thead>
            <tbody>{pred_rows if pred_rows else '<tr><td colspan="5" style="text-align:center">No Data</td></tr>'}</tbody>
        </table>
        
        <p class="updated">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} JST</p>
    </div>
    
    <script>
        const ctx = document.getElementById('performanceChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {labels},
                datasets: [
                    {{
                        label: 'Antigravity (Normal)',
                        data: {eq_opt},
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        tension: 0.3,
                        borderWidth: 3
                    }},
                    {{
                        label: 'Raptor (Wide)',
                        data: {eq_rap},
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        tension: 0.3,
                        borderWidth: 3
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#eee', font: {{ size: 14 }} }} }},
                    title: {{ display: true, text: 'Equity Curve Comparison', color: '#888' }}
                }},
                scales: {{
                    y: {{ grid: {{ color: 'rgba(255,255,255,0.1)' }}, ticks: {{ color: '#888' }} }},
                    x: {{ display: false }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

def main():
    data = load_data()
    html = generate_html(data)
    with open(DashboardConfig.OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✅ Dashboard Generated: {DashboardConfig.OUTPUT_FILE}")

if __name__ == "__main__":
    main()
