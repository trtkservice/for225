#!/usr/bin/env python3
"""
Nikkei 225 Trading Bot - AI-Powered Entry Signal Generator
Fetches market data, analyzes with Gemini AI, and generates trading signals.
"""

import os
import json
import sys
from datetime import datetime, timedelta
import pytz

# Market data
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    os.system("pip install yfinance")
    import yfinance as yf

# Gemini AI
try:
    import google.generativeai as genai
except ImportError:
    print("Installing google-generativeai...")
    os.system("pip install google-generativeai")
    import google.generativeai as genai

# Constants
JST = pytz.timezone('Asia/Tokyo')
DATA_FILE = "data/predictions.json"
SHADOW_CAPITAL = 100000  # 10万円の仮想資金
POSITION_SIZE = 1  # マイクロ先物1枚

# Tickers
TICKERS = {
    "nikkei_futures": "NKD=F",  # Nikkei 225 Futures
    "nikkei_index": "^N225",     # Nikkei 225 Index
    "sp500": "^GSPC",            # S&P 500
    "dow": "^DJI",               # Dow Jones
    "vix": "^VIX",               # VIX
    "usdjpy": "JPY=X",           # USD/JPY
    "us10y": "^TNX",             # US 10Y Treasury
}


def fetch_market_data():
    """Fetch latest market data from Yahoo Finance."""
    data = {}
    
    for name, ticker in TICKERS.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if not hist.empty:
                latest = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1]
                
                close = float(latest['Close'])
                prev_close = float(prev['Close'])
                change_pct = ((close - prev_close) / prev_close) * 100 if prev_close != 0 else 0
                
                data[name] = {
                    "close": round(close, 2),
                    "prev_close": round(prev_close, 2),
                    "change_pct": round(change_pct, 2)
                }
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            data[name] = {"close": None, "prev_close": None, "change_pct": None}
    
    return data


def create_analysis_prompt(market_data):
    """Create a prompt for Gemini to analyze market conditions."""
    
    prompt = f"""あなたは日経225先物のデイトレーダーです。以下の市場データを分析し、明日の日経225の方向を判定してください。

## 市場データ（最新）

| 指標 | 現在値 | 前日比 |
|:---|---:|---:|
| 日経225 | {market_data.get('nikkei_index', {}).get('close', 'N/A')} | {market_data.get('nikkei_index', {}).get('change_pct', 'N/A')}% |
| S&P 500 | {market_data.get('sp500', {}).get('close', 'N/A')} | {market_data.get('sp500', {}).get('change_pct', 'N/A')}% |
| NYダウ | {market_data.get('dow', {}).get('close', 'N/A')} | {market_data.get('dow', {}).get('change_pct', 'N/A')}% |
| VIX | {market_data.get('vix', {}).get('close', 'N/A')} | {market_data.get('vix', {}).get('change_pct', 'N/A')}% |
| ドル円 | {market_data.get('usdjpy', {}).get('close', 'N/A')} | {market_data.get('usdjpy', {}).get('change_pct', 'N/A')}% |
| 米10年債利回り | {market_data.get('us10y', {}).get('close', 'N/A')} | {market_data.get('us10y', {}).get('change_pct', 'N/A')}% |

## 判定ルール

以下のルールに基づいて判定してください：

### LONG（買い）条件
- 米国株（S&P500 or ダウ）が+0.3%以上 AND VIX < 20
- または、ドル円が+0.3%以上（円安）AND VIX < 25

### SHORT（売り）条件
- 米国株（S&P500 or ダウ）が-0.5%以下 OR VIX > 25
- または、ドル円が-0.5%以下（円高）

### WAIT（様子見）条件
- 上記のどちらにも該当しない場合

## 出力形式

以下のJSON形式で出力してください。他の文章は不要です。

```json
{{
  "direction": "LONG" または "SHORT" または "WAIT",
  "confidence": "HIGH" または "MEDIUM" または "LOW",
  "stop_points": 200,
  "target_points": 400,
  "reasoning": "判定理由を1-2文で"
}}
```
"""
    return prompt


def call_gemini(prompt, api_key):
    """Call Gemini API for analysis."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    response = model.generate_content(prompt)
    return response.text


def parse_ai_response(response_text):
    """Parse AI response to extract prediction."""
    try:
        # Extract JSON from response
        import re
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"Error parsing AI response: {e}")
    
    return {
        "direction": "WAIT",
        "confidence": "LOW",
        "stop_points": 200,
        "target_points": 400,
        "reasoning": "Failed to parse AI response"
    }


def load_predictions():
    """Load existing predictions from file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"predictions": [], "shadow_portfolio": {"capital": SHADOW_CAPITAL, "position": None, "trades": []}}


def save_predictions(data):
    """Save predictions to file."""
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_shadow_portfolio(data, prediction, market_data):
    """Update shadow portfolio with new prediction and evaluate previous trades."""
    portfolio = data.get("shadow_portfolio", {"capital": SHADOW_CAPITAL, "position": None, "trades": []})
    
    nikkei_price = market_data.get('nikkei_index', {}).get('close')
    
    # Check if previous position hit stop or target
    if portfolio.get("position"):
        pos = portfolio["position"]
        entry = pos["entry_price"]
        direction = pos["direction"]
        stop = pos["stop"]
        target = pos["target"]
        
        if nikkei_price:
            pnl_points = (nikkei_price - entry) if direction == "LONG" else (entry - nikkei_price)
            pnl_yen = pnl_points * 10  # マイクロ先物: 1円 = 10円
            
            # Check stop/target
            closed = False
            close_reason = ""
            
            if direction == "LONG":
                if nikkei_price <= stop:
                    closed = True
                    close_reason = "STOP"
                    pnl_points = stop - entry
                elif nikkei_price >= target:
                    closed = True
                    close_reason = "TARGET"
                    pnl_points = target - entry
            else:  # SHORT
                if nikkei_price >= stop:
                    closed = True
                    close_reason = "STOP"
                    pnl_points = entry - stop
                elif nikkei_price <= target:
                    closed = True
                    close_reason = "TARGET"
                    pnl_points = entry - target
            
            if closed:
                pnl_yen = pnl_points * 10
                portfolio["capital"] += pnl_yen
                portfolio["trades"].append({
                    "entry_date": pos["entry_date"],
                    "exit_date": datetime.now(JST).strftime("%Y-%m-%d %H:%M"),
                    "direction": direction,
                    "entry_price": entry,
                    "exit_price": nikkei_price,
                    "pnl_points": round(pnl_points, 0),
                    "pnl_yen": round(pnl_yen, 0),
                    "close_reason": close_reason
                })
                portfolio["position"] = None
    
    # Open new position if signal is LONG or SHORT
    if prediction["direction"] in ["LONG", "SHORT"] and not portfolio.get("position") and nikkei_price:
        stop_price = nikkei_price - prediction["stop_points"] if prediction["direction"] == "LONG" else nikkei_price + prediction["stop_points"]
        target_price = nikkei_price + prediction["target_points"] if prediction["direction"] == "LONG" else nikkei_price - prediction["target_points"]
        
        portfolio["position"] = {
            "direction": prediction["direction"],
            "entry_date": datetime.now(JST).strftime("%Y-%m-%d %H:%M"),
            "entry_price": nikkei_price,
            "stop": stop_price,
            "target": target_price,
            "size": POSITION_SIZE
        }
    
    data["shadow_portfolio"] = portfolio
    return data


def calculate_stats(data):
    """Calculate trading statistics."""
    trades = data.get("shadow_portfolio", {}).get("trades", [])
    if not trades:
        return {"total_trades": 0, "win_rate": 0, "total_pnl": 0}
    
    wins = len([t for t in trades if t["pnl_yen"] > 0])
    total_pnl = sum(t["pnl_yen"] for t in trades)
    
    return {
        "total_trades": len(trades),
        "win_rate": round(wins / len(trades) * 100, 1) if trades else 0,
        "total_pnl": total_pnl,
        "capital": data.get("shadow_portfolio", {}).get("capital", SHADOW_CAPITAL)
    }


def main():
    print("=" * 60)
    print("🤖 Nikkei 225 Trading Bot")
    print(f"📅 {datetime.now(JST).strftime('%Y-%m-%d %H:%M JST')}")
    print("=" * 60)
    
    # Get API key from environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        sys.exit(1)
    
    # Fetch market data
    print("\n📊 Fetching market data...")
    market_data = fetch_market_data()
    
    for name, values in market_data.items():
        if values.get("close"):
            print(f"   {name}: {values['close']} ({values['change_pct']:+.2f}%)")
    
    # Create prompt and call Gemini
    print("\n🧠 Analyzing with Gemini AI...")
    prompt = create_analysis_prompt(market_data)
    response = call_gemini(prompt, api_key)
    prediction = parse_ai_response(response)
    
    print(f"\n📈 Prediction: {prediction['direction']}")
    print(f"   Confidence: {prediction['confidence']}")
    print(f"   Stop: {prediction['stop_points']} points")
    print(f"   Target: {prediction['target_points']} points")
    print(f"   Reason: {prediction['reasoning']}")
    
    # Load and update data
    data = load_predictions()
    
    # Add prediction to history
    data["predictions"].append({
        "timestamp": datetime.now(JST).strftime("%Y-%m-%d %H:%M"),
        "market_data": market_data,
        "prediction": prediction
    })
    
    # Update shadow portfolio
    data = update_shadow_portfolio(data, prediction, market_data)
    
    # Calculate stats
    stats = calculate_stats(data)
    
    print(f"\n💰 Shadow Portfolio Status:")
    print(f"   Capital: ¥{stats['capital']:,.0f}")
    print(f"   Total Trades: {stats['total_trades']}")
    print(f"   Win Rate: {stats['win_rate']}%")
    print(f"   Total P&L: ¥{stats['total_pnl']:+,.0f}")
    
    if data["shadow_portfolio"].get("position"):
        pos = data["shadow_portfolio"]["position"]
        print(f"\n📍 Current Position:")
        print(f"   Direction: {pos['direction']}")
        print(f"   Entry: {pos['entry_price']}")
        print(f"   Stop: {pos['stop']}")
        print(f"   Target: {pos['target']}")
    
    # Save data
    save_predictions(data)
    print(f"\n✅ Results saved to {DATA_FILE}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
