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
import pandas as pd

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


    return data


def calculate_atr(hist, period=5):
    """Calculate Average True Range (ATR)."""
    try:
        high_low = hist['High'] - hist['Low']
        high_close = (hist['High'] - hist['Close'].shift()).abs()
        low_close = (hist['Low'] - hist['Close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        atr = true_range.rolling(window=period).mean().iloc[-1]
        return round(atr, 2)
    except Exception:
        return 400.0  # Fallback to default


def fetch_market_data():
    """Fetch latest market data from Yahoo Finance."""
    data = {}
    import pandas as pd  # Import locally to avoid global if not needed
    
    for name, ticker in TICKERS.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="10d")  # Need more days for ATR
            if not hist.empty:
                latest = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1]
                
                close = float(latest['Close'])
                prev_close = float(prev['Close'])
                change_pct = ((close - prev_close) / prev_close) * 100 if prev_close != 0 else 0
                
                # Calculate ATR for Nikkei Index
                atr = None
                if name == "nikkei_index":
                    atr = calculate_atr(hist)
                
                data[name] = {
                    "close": round(close, 2),
                    "high": round(float(latest['High']), 2),
                    "low": round(float(latest['Low']), 2),
                    "prev_close": round(prev_close, 2),
                    "change_pct": round(change_pct, 2),
                    "atr": atr
                }
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            data[name] = {"close": None, "prev_close": None, "change_pct": None}
    
    return data


def python_rule_check(market_data):
    """
    Layer 1: Python-based rule check (deterministic)
    Returns: (signal, strength, reason)
    """
    # Extract values safely
    sp500_change = market_data.get('sp500', {}).get('change_pct', 0) or 0
    dow_change = market_data.get('dow', {}).get('change_pct', 0) or 0
    vix = market_data.get('vix', {}).get('close', 20) or 20
    usdjpy_change = market_data.get('usdjpy', {}).get('change_pct', 0) or 0
    us10y_change = market_data.get('us10y', {}).get('change_pct', 0) or 0
    
    # Strong LONG conditions
    if (sp500_change > 0.5 or dow_change > 0.5) and vix < 18:
        return "LONG", "STRONG", "米国株大幅上昇 + VIX低水準"
    
    # Medium LONG conditions
    if (sp500_change > 0.3 or dow_change > 0.3) and vix < 20:
        if usdjpy_change > 0.3:  # 円安も追い風
            return "LONG", "STRONG", "米国株上昇 + 円安 + VIX安定"
        return "LONG", "MEDIUM", "米国株上昇 + VIX安定"
    
    # Weak LONG (円安のみ)
    if usdjpy_change > 0.5 and vix < 22:
        return "LONG", "WEAK", "円安トレンド"
    
    # Strong SHORT conditions
    if vix > 25:
        return "SHORT", "STRONG", "VIX急騰（リスクオフ）"
    
    if sp500_change < -0.5 or dow_change < -0.5:
        return "SHORT", "STRONG", "米国株大幅下落"
    
    # Medium SHORT conditions
    if (sp500_change < -0.3 or dow_change < -0.3) and vix > 20:
        return "SHORT", "MEDIUM", "米国株下落 + VIX上昇"
    
    if usdjpy_change < -0.5:  # 急激な円高
        return "SHORT", "MEDIUM", "急激な円高"
    
    # No clear signal
    return "WAIT", None, "明確なシグナルなし"

def create_confirmation_prompt(market_data, rule_signal, rule_strength, rule_reason):
    """
    Layer 2: Create a prompt for Gemini to CONFIRM or REJECT the Python rule decision.
    LLM acts as a safety filter, not the primary decision maker.
    """
    
    prompt = f"""あなたは日経225先物のリスク管理担当です。
Pythonルールが以下の判定を出しました。この判定が妥当かどうか確認してください。

## Pythonルールの判定
- **シグナル**: {rule_signal}
- **強度**: {rule_strength}
- **理由**: {rule_reason}

## 市場データ（最新）

| 指標 | 現在値 | 前日比 |
|:---|---:|---:|
| 日経225 | {market_data.get('nikkei_index', {}).get('close', 'N/A')} | {market_data.get('nikkei_index', {}).get('change_pct', 'N/A')}% |
| S&P 500 | {market_data.get('sp500', {}).get('close', 'N/A')} | {market_data.get('sp500', {}).get('change_pct', 'N/A')}% |
| NYダウ | {market_data.get('dow', {}).get('close', 'N/A')} | {market_data.get('dow', {}).get('change_pct', 'N/A')}% |
| VIX | {market_data.get('vix', {}).get('close', 'N/A')} | {market_data.get('vix', {}).get('change_pct', 'N/A')}% |
| ドル円 | {market_data.get('usdjpy', {}).get('close', 'N/A')} | {market_data.get('usdjpy', {}).get('change_pct', 'N/A')}% |
| 米10年債利回り | {market_data.get('us10y', {}).get('close', 'N/A')} | {market_data.get('us10y', {}).get('change_pct', 'N/A')}% |

## あなたのタスク

1. Pythonルールの判定に重大な問題がないか確認
2. 以下のリスクがないかチェック:
   - 重要経済指標発表（雇用統計、FOMC等）の直前
   - 市場の過熱感（連続上昇後の反落リスク等）
   - 通常とは異なる市場環境

## 出力形式（JSON）

```json
{{
  "approved": true または false,
  "final_direction": "{rule_signal}" または "WAIT",
  "confidence": "HIGH" または "MEDIUM" または "LOW",
  "stop_points": 200,
  "target_points": 400,
  "reasoning": "確認結果と補足情報"
}}
```

注意: 
- approved=true なら final_direction は Pythonルールの判定をそのまま採用
- approved=false なら final_direction は "WAIT" にして理由を説明
- 迷った場合は approved=true（Pythonルールを尊重）
"""
    return prompt


def call_gemini(prompt, api_key):
    """Call Gemini API for analysis."""
    genai.configure(api_key=api_key)
    
    # Use gemini-flash-latest as confirmed available
    model_name = 'gemini-flash-latest'
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"⚠️ Error with model {model_name}: {e}")
        print("📋 Listing available models:")
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(f"   - {m.name}")
            
            # Fallback to gemini-pro-latest
            print("🔄 Falling back to 'gemini-pro-latest'...")
            model = genai.GenerativeModel('gemini-pro-latest')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e2:
            print(f"❌ Fallback failed: {e2}")
            return None


def parse_ai_response(response_text, rule_signal="WAIT"):
    """Parse AI response to extract prediction (confirmation format)."""
    default_response = {
        "direction": rule_signal,
        "confidence": "MEDIUM",
        "stop_points": 200,
        "target_points": 400,
        "reasoning": "AI確認スキップ（Pythonルール採用）",
        "approved": True
    }
    
    if not response_text:
        return default_response
        
    try:
        # Extract JSON from response
        import re
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            
            # Convert new format to standard format
            return {
                "direction": parsed.get("final_direction", rule_signal),
                "confidence": parsed.get("confidence", "MEDIUM"),
                "stop_points": parsed.get("stop_points", 200),
                "target_points": parsed.get("target_points", 400),
                "reasoning": parsed.get("reasoning", ""),
                "approved": parsed.get("approved", True)
            }
    except Exception as e:
        print(f"Error parsing AI response: {e}")
    
    return default_response


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
    
    nikkei_data = market_data.get('nikkei_index', {})
    nikkei_price = nikkei_data.get('close')
    nikkei_high = nikkei_data.get('high')
    nikkei_low = nikkei_data.get('low')
    
    # Check if previous position hit stop or target
    if portfolio.get("position"):
        pos = portfolio["position"]
        entry = pos["entry_price"]
        direction = pos["direction"]
        stop = pos["stop"]
        target = pos["target"]
        
        if nikkei_price and nikkei_high and nikkei_low:
            pnl_points = (nikkei_price - entry) if direction == "LONG" else (entry - nikkei_price)
            
            # Check stop/target using High/Low
            closed = False
            close_reason = ""
            exit_price = nikkei_price
            
            if direction == "LONG":
                # Low triggers Stop, High triggers Target
                if nikkei_low <= stop:
                    closed = True
                    close_reason = "STOP"
                    exit_price = stop
                    pnl_points = stop - entry
                elif nikkei_high >= target:
                    closed = True
                    close_reason = "TARGET"
                    exit_price = target
                    pnl_points = target - entry
            else:  # SHORT
                # High triggers Stop, Low triggers Target
                if nikkei_high >= stop:
                    closed = True
                    close_reason = "STOP"
                    exit_price = stop
                    pnl_points = entry - stop
                elif nikkei_low <= target:
                    closed = True
                    close_reason = "TARGET"
                    exit_price = target
                    pnl_points = entry - target
            
            if closed:
                pnl_yen = pnl_points * 10
                portfolio["capital"] += pnl_yen
                portfolio["trades"].append({
                    "entry_date": pos["entry_date"],
                    "exit_date": datetime.now(JST).strftime("%Y-%m-%d %H:%M"),
                    "direction": direction,
                    "entry_price": entry,
                    "exit_price": exit_price,
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
    portfolio = data.get("shadow_portfolio", {"capital": SHADOW_CAPITAL, "trades": []})
    trades = portfolio.get("trades", [])
    capital = portfolio.get("capital", SHADOW_CAPITAL)
    
    if not trades:
        return {"total_trades": 0, "win_rate": 0, "total_pnl": 0, "capital": capital}
    
    wins = len([t for t in trades if t["pnl_yen"] > 0])
    total_pnl = sum(t["pnl_yen"] for t in trades)
    
    return {
        "total_trades": len(trades),
        "win_rate": round(wins / len(trades) * 100, 1) if trades else 0,
        "total_pnl": total_pnl,
        "capital": capital
    }


def main():
    print("=" * 60)
    print("🤖 Nikkei 225 Trading Bot (Hybrid Mode)")
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
    
    # Layer 1: Python Rule Check
    print("\n🔧 Layer 1: Python Rule Check...")
    rule_signal, rule_strength, rule_reason = python_rule_check(market_data)
    print(f"   Signal: {rule_signal}")
    print(f"   Strength: {rule_strength}")
    print(f"   Reason: {rule_reason}")
    
    # Layer 2: LLM Confirmation (only if Python has a signal)
    if rule_signal != "WAIT":
        print("\n🧠 Layer 2: Gemini AI Confirmation...")
        prompt = create_confirmation_prompt(market_data, rule_signal, rule_strength, rule_reason)
        response = call_gemini(prompt, api_key)
        prediction = parse_ai_response(response, rule_signal)
        
        if prediction.get("approved", True):
            print(f"   ✅ AI Approved: {prediction['direction']}")
        else:
            print(f"   ⚠️ AI Rejected → WAIT")
            prediction["direction"] = "WAIT"
    else:
        # No signal from Python rules, skip LLM
        print("\n⏸️ Layer 2: Skipped (No Python signal)")
        prediction = {
            "direction": "WAIT",
            "confidence": "N/A",
            "stop_points": 200,
            "target_points": 400,
            "reasoning": rule_reason,
            "approved": True
        }
    
    # Override Stop/Target with ATR based calculation
    atr = market_data.get('nikkei_index', {}).get('atr', 400) or 400
    stop_points = int(atr * 0.5)
    target_points = int(atr * 1.0)
    
    prediction['stop_points'] = stop_points
    prediction['target_points'] = target_points
    
    print(f"\n📈 Final Prediction: {prediction['direction']}")
    print(f"   Confidence: {prediction['confidence']}")
    print(f"   ATR: {atr:.0f} points")
    print(f"   Stop: {prediction['stop_points']} points (ATR x 0.5)")
    print(f"   Target: {prediction['target_points']} points (ATR x 1.0)")
    print(f"   Reason: {prediction['reasoning']}")
    
    # Load and update data
    data = load_predictions()
    
    # Add prediction to history
    data["predictions"].append({
        "timestamp": datetime.now(JST).strftime("%Y-%m-%d %H:%M"),
        "market_data": market_data,
        "rule_check": {
            "signal": rule_signal,
            "strength": rule_strength,
            "reason": rule_reason
        },
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
