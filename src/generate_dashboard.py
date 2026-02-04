#!/usr/bin/env python3
"""
Generate HTML Dashboard from predictions.json
"""

import json
import os
from datetime import datetime
import html

DATA_FILE = "data/predictions.json"
OUTPUT_FILE = "data/dashboard.html"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"predictions": [], "shadow_portfolio": {"capital": 100000, "position": None, "trades": []}}

def generate_html(data):
    portfolio = data.get("shadow_portfolio", {})
    predictions = data.get("predictions", [])
    trades = portfolio.get("trades", [])
    
    capital = portfolio.get("capital", 100000)
    initial_capital = 100000
    total_pnl = capital - initial_capital
    pnl_pct = (total_pnl / initial_capital) * 100
    
    win_trades = [t for t in trades if t.get("pnl_yen", 0) > 0]
    win_rate = (len(win_trades) / len(trades) * 100) if trades else 0
    
    position = portfolio.get("position")
    
    # Generate predictions table rows
    pred_rows = ""
    for p in reversed(predictions[-20:]):  # Last 20 predictions
        ts = p.get("timestamp", "N/A")
        scores = p.get("scores", {})
        pred = p.get("prediction", {})
        
        direction = pred.get("direction", "WAIT")
        direction_class = "long" if direction == "LONG" else "short" if direction == "SHORT" else "wait"
        
        # Format scores if available
        score_html = "-"
        if scores:
            total = scores.get("total_score", 0)
            t_score = scores.get("trend_score", 0)
            m_score = scores.get("momentum_score", 0)
            v_score = scores.get("volatility_score", 0)
            
            total_class = "profit" if total > 0 else "loss" if total < 0 else "neutral"
            
            score_html = f"""
            <div class="score-grid">
                <span class="score-item" title="Trend">T: <b class="{ 'profit' if t_score>0 else 'loss' }">{t_score:+.2f}</b></span>
                <span class="score-item" title="Momentum">M: <b class="{ 'profit' if m_score>0 else 'loss' }">{m_score:+.2f}</b></span>
                <span class="score-item" title="Volatility">V: <b class="{ 'profit' if v_score>0 else 'loss' }">{v_score:+.2f}</b></span>
                <span class="score-total">Total: <b class="{total_class}">{total:+.2f}</b></span>
            </div>
            """
        else:
            # Fallback for old data
            rule = p.get("rule_check", {})
            score_html = f"Old Rule: {rule.get('strength', '-')}"

        reasoning = html.escape(pred.get('reasoning', 'N/A'))
        pred_rows += f"""
        <tr>
            <td>{ts}</td>
            <td><span class="signal {direction_class}">{direction}</span></td>
            <td>{score_html}</td>
            <td>{pred.get('confidence', 'N/A')}</td>
            <td class="reason-cell">{reasoning}</td>
        </tr>
        """
    
    # Generate trades table rows
    trade_rows = ""
    for t in reversed(trades[-10:]):  # Last 10 trades
        pnl = t.get("pnl_yen", 0)
        pnl_class = "profit" if pnl > 0 else "loss"
        
        # Unified Reason Field
        close_reason = t.get('close_reason') or t.get('reason') or 'N/A'
        
        trade_rows += f"""
        <tr>
            <td>{t.get('entry_date', 'N/A')}</td>
            <td>{t.get('exit_date', 'N/A')}</td>
            <td>{t.get('direction', 'N/A')}</td>
            <td>{t.get('entry_price', 0):,.0f}</td>
            <td>{t.get('exit_price', 0):,.0f}</td>
            <td class="{pnl_class}">¥{pnl:+,.0f}</td>
            <td>{close_reason}</td>
        </tr>
        """
    
    # Capital history for chart
    capital_history = [initial_capital]
    for t in trades:
        capital_history.append(capital_history[-1] + t.get("pnl_yen", 0))
    capital_labels = list(range(len(capital_history)))
    
    # Current position display
    position_html = ""
    if position:
        pos_direction = position.get("direction", "N/A")
        pos_class = "long" if pos_direction == "LONG" else "short"
        position_html = f"""
        <div class="position-card {pos_class}">
            <h3>📍 Current Position</h3>
            <div class="position-details">
                <span class="signal {pos_class}">{pos_direction}</span>
                <p>Entry: {position.get('entry_price', 0):,.0f}</p>
                <p>Stop: {position.get('stop', 0):,.0f}</p>
                <p>Target: {position.get('target', 0):,.0f}</p>
                <p>Since: {position.get('entry_date', 'N/A')}</p>
            </div>
        </div>
        """
    else:
        position_html = """
        <div class="position-card neutral">
            <h3>📍 Current Position</h3>
            <p>No open position</p>
        </div>
        """
    
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nikkei 225 Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 30px;
            font-size: 2em;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-card h3 {{
            font-size: 0.9em;
            color: #888;
            margin-bottom: 10px;
        }}
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .profit {{ color: #00d4ff; }}
        .loss {{ color: #ff6b6b; }}
        .neutral {{ color: #888; }}
        .chart-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            overflow: hidden;
        }}
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{
            background: rgba(0,0,0,0.3);
            font-weight: 600;
        }}
        .signal {{
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.85em;
        }}
        .signal.long {{
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
        }}
        .signal.short {{
            background: rgba(255, 107, 107, 0.2);
            color: #ff6b6b;
        }}
        .signal.wait {{
            background: rgba(136, 136, 136, 0.2);
            color: #888;
        }}
        .position-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .position-card.long {{
            border-color: rgba(0, 212, 255, 0.5);
        }}
        .position-card.short {{
            border-color: rgba(255, 107, 107, 0.5);
        }}
        .section-title {{
            margin: 30px 0 20px;
            font-size: 1.3em;
        }}
        .updated {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            font-size: 0.9em;
        }}
        .score-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 5px;
            font-size: 0.8em;
            width: 160px;
        }}
        .score-item {{
            color: #aaa;
        }}
        .score-total {{
            grid-column: span 2;
            border-top: 1px solid rgba(255,255,255,0.1);
            margin-top: 5px;
            padding-top: 2px;
            text-align: center;
        }}
        .reason-cell {{
            font-size: 0.85em;
            color: #ccc;
            max-width: 400px;
            line-height: 1.4;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Antigravity Trading Dashboard</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>💰 Capital</h3>
                <div class="value">¥{capital:,.0f}</div>
            </div>
            <div class="stat-card">
                <h3>📊 Total P&L</h3>
                <div class="value {'profit' if total_pnl >= 0 else 'loss'}">¥{total_pnl:+,.0f}</div>
            </div>
            <div class="stat-card">
                <h3>📈 Return</h3>
                <div class="value {'profit' if pnl_pct >= 0 else 'loss'}">{pnl_pct:+.2f}%</div>
            </div>
            <div class="stat-card">
                <h3>🎯 Win Rate</h3>
                <div class="value">{win_rate:.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>📝 Total Trades</h3>
                <div class="value">{len(trades)}</div>
            </div>
            <div class="stat-card">
                <h3>🔮 Predictions</h3>
                <div class="value">{len(predictions)}</div>
            </div>
        </div>
        
        {position_html}
        
        <div class="chart-container">
            <canvas id="capitalChart"></canvas>
        </div>
        
        <h2 class="section-title">📋 Recent Predictions</h2>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Signal</th>
                    <th>Scores (Antigravity)</th>
                    <th>Confidence</th>
                    <th>Reasoning</th>
                </tr>
            </thead>
            <tbody>
                {pred_rows if pred_rows else '<tr><td colspan="5">No predictions yet</td></tr>'}
            </tbody>
        </table>
        
        <h2 class="section-title">💹 Trade History</h2>
        <table>
            <thead>
                <tr>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>Direction</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>P&L</th>
                    <th>Reason</th>
                </tr>
            </thead>
            <tbody>
                {trade_rows if trade_rows else '<tr><td colspan="7">No trades yet</td></tr>'}
            </tbody>
        </table>
        
        <p class="updated">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    </div>
    
    <script>
        const ctx = document.getElementById('capitalChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {capital_labels},
                datasets: [{{
                    label: 'Capital (¥)',
                    data: {capital_history},
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    fill: true,
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        labels: {{ color: '#eee' }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Trade #', color: '#888' }},
                        ticks: {{ color: '#888' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }},
                    y: {{
                        title: {{ display: true, text: 'Capital (¥)', color: '#888' }},
                        ticks: {{ color: '#888' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    return html

def main():
    data = load_data()
    html = generate_html(data)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Dashboard generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
