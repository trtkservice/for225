#!/usr/bin/env python3
"""
Nikkei 225 Trading Bot - Antigravity Engine (Advanced Technical Analysis)
"""

import os
import json
import sys
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np

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
SHADOW_CAPITAL = 100000
POSITION_SIZE = 1

# Tickers
TICKERS = {
    "nikkei_futures": "NKD=F",
    "nikkei_index": "^N225",
    "sp500": "^GSPC",
    "dow": "^DJI",
    "vix": "^VIX",
    "usdjpy": "JPY=X",
    "us10y": "^TNX",
}

def fetch_data():
    """Fetch both Daily and Intraday data."""
    data = {}
    
    # 1. Fetch Daily Data (for Trend/DEEP) - Last 1 year
    for name, ticker in TICKERS.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1y")
            data[f"{name}_daily"] = hist
        except Exception as e:
            print(f"Error fetching daily {name}: {e}")
            data[f"{name}_daily"] = pd.DataFrame()

    # 2. Fetch Intraday Data (for Momentum/FAST) - Last 5 days, 15m intervals
    # Only for Nikkei Futures to gauge momentum
    try:
        t = yf.Ticker(TICKERS["nikkei_futures"])
        hist_15m = t.history(period="5d", interval="15m")
        data["nikkei_15m"] = hist_15m
    except Exception as e:
        print(f"Error fetching intraday Nikkei: {e}")
        data["nikkei_15m"] = pd.DataFrame()

    return data

# --- Technical Analysis Library ---

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def calc_bollinger(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calc_atr(hist, period=14):
    high_low = hist['High'] - hist['Low']
    high_close = (hist['High'] - hist['Close'].shift()).abs()
    low_close = (hist['Low'] - hist['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

# --- Scoring Engine ---

def calculate_antigravity_score(data):
    """
    Calculate integrated score (-1.0 to +1.0)
    Layer 1: Trend (Daily)
    Layer 2: Momentum (15m)
    Layer 3: Volatility/Risk
    """
    scores = {
        "trend_score": 0,
        "momentum_score": 0,
        "volatility_score": 0,
        "total_score": 0,
        "details": {}
    }
    
    nikkei_daily = data.get("nikkei_futures_daily")
    nikkei_15m = data.get("nikkei_15m")
    vix_daily = data.get("vix_daily")
    
    if nikkei_daily is None or nikkei_daily.empty:
        return scores

    # --- Layer 1: Trend (DEEP) ---
    # EMA Analysis
    close = nikkei_daily['Close']
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
    current_price = close.iloc[-1]
    
    trend_val = 0
    if current_price > ema20: trend_val += 0.3
    if current_price > ema50: trend_val += 0.3
    if current_price > ema200: trend_val += 0.4
    if ema20 > ema50: trend_val += 0.2  # Golden Cross state
    
    # Normalize to -1.0 to 1.0 (approx)
    if current_price < ema20: trend_val -= 0.3
    if current_price < ema50: trend_val -= 0.3
    if current_price < ema200: trend_val -= 0.4
    if ema20 < ema50: trend_val -= 0.2
    
    scores["trend_score"] = round(np.clip(trend_val, -1.0, 1.0), 3)
    scores["details"]["ema_alignment"] = "Bullish" if trend_val > 0 else "Bearish"

    # --- Layer 2: Momentum (FAST) ---
    mom_val = 0
    if nikkei_15m is not None and not nikkei_15m.empty:
        close_15m = nikkei_15m['Close']
        
        # RSI (14)
        rsi = calc_rsi(close_15m).iloc[-1]
        scores["details"]["rsi_15m"] = round(rsi, 1)
        
        if rsi > 70:
            mom_val -= 0.3  # Overbought, risk of pullback
        elif rsi < 30:
            mom_val += 0.3  # Oversold, bounce candidate
        elif 50 < rsi <= 70:
            mom_val += 0.2  # Strong momentum
        elif 30 <= rsi < 50:
            mom_val -= 0.2  # Weak momentum
            
        # MACD (15m)
        macd, signal, hist = calc_macd(close_15m)
        last_hist = hist.iloc[-1]
        prev_hist = hist.iloc[-2]
        
        if last_hist > 0 and last_hist > prev_hist:
            mom_val += 0.4  # Accelerating Up
        elif last_hist > 0 and last_hist < prev_hist:
            mom_val += 0.1  # Decelerating Up
        elif last_hist < 0 and last_hist < prev_hist:
            mom_val -= 0.4  # Accelerating Down
        elif last_hist < 0 and last_hist > prev_hist:
            mom_val -= 0.1  # Decelerating Down
            
    scores["momentum_score"] = round(np.clip(mom_val, -1.0, 1.0), 3)

    # --- Layer 3: Risk / Volatility ---
    vol_val = 0
    if not vix_daily.empty:
        vix = vix_daily['Close'].iloc[-1]
        scores["details"]["vix"] = round(vix, 2)
        
        if vix < 15: vol_val += 0.2     # Risk On
        elif vix < 20: vol_val += 0.0
        elif vix > 30: vol_val -= 0.8   # Extreme Panic
        elif vix > 20: vol_val -= 0.3   # Caution
        
    scores["volatility_score"] = round(vol_val, 3)
    
    # --- Total Integration ---
    # Weight: Trend 40%, Momentum 40%, Volatility 20%
    total = (scores["trend_score"] * 0.4) + (scores["momentum_score"] * 0.4) + (scores["volatility_score"] * 0.2)
    scores["total_score"] = round(total, 3)
    
    # Signal Generation
    if total > 0.3: scores["signal"] = "LONG"
    elif total < -0.3: scores["signal"] = "SHORT"
    else: scores["signal"] = "WAIT"
    
    scores["strength"] = "STRONG" if abs(total) > 0.6 else "MEDIUM" if abs(total) > 0.3 else "WEAK"
    
    return scores

# --- Gemini Prompt ---

def create_antigravity_prompt(scores, market_data):
    prompt = f"""あなたはヘッジファンドのトップトレーダーです。
高度なテクニカル分析エンジン「Antigravity」が以下の市場解析結果を出しました。
このデータを元に、最終的なエントリー判断を行ってください。

## 📊 Antigravity 解析スコア (-1.0 〜 +1.0)
| レイヤー | スコア | 評価 | 意味 |
|:---|:---|:---|:---|
| 🌊 Trend (日足) | **{scores['trend_score']}** | {scores['details'].get('ema_alignment')} | { '上昇トレンド' if scores['trend_score'] > 0 else '下降トレンド' } |
| 🚀 Momentum (15分足) | **{scores['momentum_score']}** | RSI: {scores['details'].get('rsi_15m')} | { '短期上昇圧力' if scores['momentum_score'] > 0 else '短期下落圧力' } |
| ⚠️ Volatility (VIX) | **{scores['volatility_score']}** | VIX: {scores['details'].get('vix')} | { 'リスクオン' if scores['volatility_score'] >= 0 else 'リスクオフ（警戒）' } |
| **💎 総合スコア** | **{scores['total_score']}** | **{scores['signal']}** | (基準: >0.3でBUY, <-0.3でSELL) |

## あなたのタスク
エンジンの算出結果（数学的根拠）を、ファンダメンタルズや市場心理の観点から**最終チェック**してください。

1. **スコアの整合性確認**: 短期と長期のスコアが矛盾している場合（例: Trend↑ Momentum↓）、どう判断するか？（通常はTrend優先だが、Momentumが極端な場合は反転の兆しか？）
2. **ブラックスワン回避**: 突発的なリスク要因がないか考慮（プロンプト等には含まれませんが、一般的な市場知識として「通常の動きか」を判断）

## 出力形式（JSON）
```json
{{
  "approved": true,
  "final_direction": "{scores['signal']}",
  "reasoning": "Trendスコアが強く、15分足のRSIも過熱感がないため、上値余地ありと判断。VIXも安定している。",
  "risk_alert": "なし"
}}
```
"""
    return prompt

# --- Main Functions ---

def call_gemini(prompt, api_key):
    genai.configure(api_key=api_key)
    model_name = 'gemini-flash-latest'
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"⚠️ API Error: {e}")
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except:
            return None

def parse_ai_response(response_text, rule_signal="WAIT"):
    default = {"direction": rule_signal, "confidence": "MEDIUM", "approved": True, "reasoning": "Fallback"}
    if not response_text: return default
    try:
        import re
        match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            return {
                "direction": parsed.get("final_direction", rule_signal),
                "confidence": "HIGH" if parsed.get("approved") else "LOW",
                "reasoning": parsed.get("reasoning", ""),
                "approved": parsed.get("approved", True)
            }
    except Exception:
        pass
    return default

def load_predictions():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    return {"predictions": [], "shadow_portfolio": {"capital": SHADOW_CAPITAL, "position": None, "trades": []}}

def save_predictions(data):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)

def update_shadow_portfolio(data, prediction, market_data, atr_val):
    portfolio = data.get("shadow_portfolio", {"capital": SHADOW_CAPITAL, "position": None, "trades": []})
    
    # Get Prices
    nikkei = market_data.get("nikkei_futures_daily")
    if nikkei is None or nikkei.empty: return data
    
    last_row = nikkei.iloc[-1]
    price = last_row['Close']
    high = last_row['High']
    low = last_row['Low']
    
    entry_price = price # Use close as "current" for simulation entry

    # Manage Open Position
    if portfolio.get("position"):
        pos = portfolio["position"]
        direction = pos["direction"]
        stop = pos["stop"]
        target = pos["target"]
        
        hit_stop = False
        hit_target = False
        
        # Conservative check
        if direction == "LONG":
            hit_stop = low <= stop
            hit_target = high >= target
        else:
            hit_stop = high >= stop
            hit_target = low <= target
            
        closed = False
        reason = ""
        exit_p = price
        
        if hit_stop:
            closed = True; reason = "STOP"; exit_p = stop
        elif hit_target:
            closed = True; reason = "TARGET"; exit_p = target
            
        if closed:
            pnl = (exit_p - pos["entry_price"]) * 10 if direction == "LONG" else (pos["entry_price"] - exit_p) * 10
            portfolio["capital"] += pnl
            portfolio["trades"].append({
                "entry_date": pos["entry_date"],
                "exit_date": datetime.now(JST).strftime("%Y-%m-%d %H:%M"),
                "direction": direction,
                "pnl_yen": int(pnl),
                "reason": reason
            })
            portfolio["position"] = None

    # Open New Position
    if prediction["direction"] in ["LONG", "SHORT"] and not portfolio.get("position"):
        stop_dist = int(atr_val * 0.5)
        target_dist = int(atr_val * 1.0)
        
        stop_price = entry_price - stop_dist if prediction["direction"] == "LONG" else entry_price + stop_dist
        target_price = entry_price + target_dist if prediction["direction"] == "LONG" else entry_price - target_dist
        
        portfolio["position"] = {
            "direction": prediction["direction"],
            "entry_date": datetime.now(JST).strftime("%Y-%m-%d %H:%M"),
            "entry_price": entry_price,
            "stop": stop_price,
            "target": target_price,
            "stop_dist": stop_dist,
            "target_dist": target_dist
        }
        
    data["shadow_portfolio"] = portfolio
    return data

def main():
    print("="*60)
    print("🚀 Antigravity Engine v2.0 Starting...")
    print(f"📅 {datetime.now(JST).strftime('%Y-%m-%d %H:%M JST')}")
    print("="*60)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ No API Key"); sys.exit(1)

    # 1. Fetch Data
    print("📥 Fetching Market Data...")
    raw_data = fetch_data()
    
    # 2. Calculate Technical Scores
    print("\n🧮 Calculating Antigravity Scores...")
    scores = calculate_antigravity_score(raw_data)
    
    print(f"   🌊 Trend Score:     {scores['trend_score']:+}")
    print(f"   🚀 Momentum Score:  {scores['momentum_score']:+}")
    print(f"   ⚠️ Volatility Score: {scores['volatility_score']:+}")
    print(f"   💎 TOTAL SCORE:      {scores['total_score']:+} ({scores['signal']} {scores['strength']})")

    # 3. LLM Confirmation
    print("\n🧠 Gemini AI Final Check...")
    prompt = create_antigravity_prompt(scores, raw_data)
    response = call_gemini(prompt, api_key)
    prediction = parse_ai_response(response, scores['signal'])
    
    print(f"   🤖 AI Verdict: {prediction['direction']} (Approved: {prediction['approved']})")
    print(f"   📝 Reason: {prediction['reasoning']}")

    # 4. ATR Calculation for Risk Management
    nikkei_d = raw_data.get("nikkei_futures_daily")
    atr_val = 400
    if nikkei_d is not None and not nikkei_d.empty:
        atr_series = calc_atr(nikkei_d)
        atr_val = atr_series.iloc[-1]
    
    print(f"   📏 ATR (14): {atr_val:.0f}")

    # 5. Save & Update
    data = load_predictions()
    
    # Record history
    data["predictions"].append({
        "timestamp": datetime.now(JST).strftime("%Y-%m-%d %H:%M"),
        "scores": scores,
        "prediction": prediction,
        "atr": round(atr_val, 2)
    })
    
    data = update_shadow_portfolio(data, prediction, raw_data, atr_val)
    save_predictions(data)
    
    # Stats
    p = data["shadow_portfolio"]
    print(f"\n💰 Capital: ¥{p['capital']:,.0f}")
    if p["position"]:
        print(f"📍 Position: {p['position']['direction']} @ {p['position']['entry_price']:.0f}")
    
    print("✅ Done.")

if __name__ == "__main__":
    main()
