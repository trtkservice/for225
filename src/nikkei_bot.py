#!/usr/bin/env python3
"""
Nikkei 225 Trading Bot - Antigravity Engine v2.1 (Refactored)
Author: Antigravity Agent
"""

import os
import json
import sys
import math
import re
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
import google.generativeai as genai

# --- Configuration & Constants ---

class Config:
    # System
    JST = pytz.timezone('Asia/Tokyo')
    DATA_FILE = "data/predictions.json"
    
    # Trading Specs (Nikkei 225 Micro)
    CONTRACT_MULTIPLIER = 10  # 1 point = 10 JPY
    TICK_SIZE = 5             # Minimum fluctuation
    COST_PER_TRADE = 50       # Estimated commission + slippage per trade
    LOTS = 1                  # Number of Contracts (Leverage)
    
    # Strategy Parameters
    SHADOW_CAPITAL = 100000
    MAX_HOLD_DAYS = 5
    
    # Strategy Definitions (A/B Testing)
    STRATEGIES = {
        "optimized": {
            "name": "Antigravity (Normal)",
            "stop_mult": 0.6,
            "target_mult": 1.2
        },
        "raptor": {
            "name": "Raptor (Wide)",
            "stop_mult": 1.2,
            "target_mult": 2.4
        }
    }
    
    # Antigravity Weights
    WEIGHT_TREND = 0.4
    WEIGHT_MOMENTUM = 0.4
    WEIGHT_VOLATILITY = 0.2
    
    # RiskGate (Gap Filter)
    GAP_THRESHOLD = 0.005  # 0.5% (approx 200 JPY) -> Skip if gap is too large
    
    # Tickers
    TICKERS = {
        "nikkei_futures": "NKD=F",
        "nikkei_index": "^N225",
        "vix": "^VIX"
    }

# --- Helper Functions ---

def round_to_tick(price: float) -> int:
    """Round price to the nearest tick size."""
    if pd.isna(price): return 0
    return int(round(price / Config.TICK_SIZE) * Config.TICK_SIZE)

# --- Components ---

class MarketDataManager:
    """Handles data fetching from Yahoo Finance."""
    
    @staticmethod
    def fetch_all():
        data = {}
        # 1. Daily Data (1y)
        for name, ticker in Config.TICKERS.items():
            try:
                t = yf.Ticker(ticker)
                data[f"{name}_daily"] = t.history(period="1y")
            except Exception as e:
                print(f"⚠️ Error fetching {name}: {e}")
                data[f"{name}_daily"] = pd.DataFrame()
        
        # 2. Intraday Data (5d, 15m) for Momentum
        try:
            t = yf.Ticker(Config.TICKERS["nikkei_futures"])
            data["nikkei_15m"] = t.history(period="5d", interval="15m")
        except Exception as e:
            print(f"⚠️ Error fetching intraday: {e}")
            data["nikkei_15m"] = pd.DataFrame()
            
        return data

class TechnicalAnalysis:
    """Statistical Calculation Library."""
    
    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> dict:
        """Detect Candlestick Patterns (Pinbar, Engulfing)."""
        if len(df) < 2: return {}
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        open_p, close_p = curr['Open'], curr['Close']
        high_p, low_p = curr['High'], curr['Low']
        
        body = abs(close_p - open_p)
        total_len = high_p - low_p
        if total_len == 0: return {}
        
        # 1. Pin Bar (Hammer / Shooting Star)
        # Lower Wick > 2 * Body (Bullish)
        # Upper Wick > 2 * Body (Bearish)
        lower_wick = min(open_p, close_p) - low_p
        upper_wick = high_p - max(open_p, close_p)
        
        is_bullish_pin = (lower_wick > 2 * body) and (upper_wick < body)
        is_bearish_pin = (upper_wick > 2 * body) and (lower_wick < body)
        
        # 2. Engulfing (Original simple logic)
        # Bullish: Prev Red, Curr Green, Curr Open < Prev Close, Curr Close > Prev Open
        is_bullish_eng = (prev['Close'] < prev['Open']) and \
                         (close_p > open_p) and \
                         (open_p <= prev['Close']) and \
                         (close_p >= prev['Open'])
                         
        is_bearish_eng = (prev['Close'] > prev['Open']) and \
                         (close_p < open_p) and \
                         (open_p >= prev['Close']) and \
                         (close_p <= prev['Open'])

        return {
            "bullish_pinbar": is_bullish_pin,
            "bearish_pinbar": is_bearish_pin,
            "bullish_engulfing": is_bullish_eng,
            "bearish_engulfing": is_bearish_eng
        }

    @staticmethod
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calc_macd(series, fast=12, slow=26, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line, macd - signal_line

    @staticmethod
    def detect_divergence(df: pd.DataFrame, period=14, window=10) -> dict:
        """Detect RSI Divergence (Reversal Signal)."""
        if len(df) < window + 2: return {}
        
        close = df['Close']
        rsi = TechnicalAnalysis.calc_rsi(close, period)
        
        # Lookback for peaks in the last 'window' days (excluding today)
        # Today
        curr_price = close.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        
        # Previous relevant peak/trough logic is complex to perfect simply.
        # We use a simplified approach: Compare today with the Min/Max of the window
        # AND check if RSI disagrees.
        
        # Window range (e.g., 5 to 1 days ago)
        past_window = df.iloc[-(window+1):-1]
        past_rsi_window = rsi.iloc[-(window+1):-1]
        
        # 1. Bearish Divergence (Price Higher High, RSI Lower High)
        max_price_idx = past_window['High'].idxmax()
        max_price = past_window['High'].max()
        max_rsi_at_peak = rsi[max_price_idx] # RSI at that price peak
        
        is_bearish_div = False
        if curr_price > max_price: # Price made new high
            if curr_rsi < max_rsi_at_peak: # RSI is lower than it was at previous peak
                is_bearish_div = True
                
        # 2. Bullish Divergence (Price Lower Low, RSI Higher Low)
        min_price_idx = past_window['Low'].idxmin()
        min_price = past_window['Low'].min()
        min_rsi_at_peak = rsi[min_price_idx]
        
        is_bullish_div = False
        if curr_price < min_price: # Price made new low
            if curr_rsi > min_rsi_at_peak: # RSI is higher than at previous low
                is_bullish_div = True
                
        return {
            "bullish_divergence": is_bullish_div,
            "bearish_divergence": is_bearish_div
        }

    @staticmethod
    def calc_atr(hist, period=14):
        high_low = hist['High'] - hist['Low']
        high_close = (hist['High'] - hist['Close'].shift()).abs()
        low_close = (hist['Low'] - hist['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()

class AntigravityEngine:
    """
    Simulates physical forces in the market.
    Trend (River), Momentum (Wind), Volatility (Temperature).
    """
    
    def __init__(self, market_data):
        self.data = market_data
        self.scores = {
            "trend": 0.0, "momentum": 0.0, "volatility": 0.0, 
            "total": 0.0, "signal": "WAIT", "strength": "WEAK",
            "details": {}
        }

    def analyze(self):
        self._analyze_trend()
        self._analyze_momentum()
        self._analyze_volatility()
        self._integrate()
        return self.scores

    def _analyze_trend(self):
        df = self.data.get("nikkei_futures_daily")
        if df is None or len(df) < 200: return

        close = df['Close']
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
        price = close.iloc[-1]
        
        val = 0
        # Bullish factors
        if price > ema20: val += 0.3
        if price > ema50: val += 0.3
        if price > ema200: val += 0.4
        if ema20 > ema50: val += 0.2
        
        # Bearish factors
        if price < ema20: val -= 0.3
        if price < ema50: val -= 0.3
        if price < ema200: val -= 0.4
        if ema20 < ema50: val -= 0.2
        
        self.scores["trend"] = round(np.clip(val, -1.0, 1.0), 3)
        self.scores["details"]["trend_summary"] = "Bullish" if val > 0 else "Bearish"

    def _analyze_momentum(self):
        df = self.data.get("nikkei_15m")
        val = 0
        if df is not None and len(df) > 30:
            close = df['Close']
            
            # RSI
            rsi = TechnicalAnalysis.calc_rsi(close).iloc[-1]
            self.scores["details"]["rsi"] = round(rsi, 1) if not np.isnan(rsi) else 50
            
            if rsi > 70: val -= 0.3
            elif rsi < 30: val += 0.3
            elif rsi > 50: val += 0.2
            else: val -= 0.2
            
            # MACD
            _, _, hist = TechnicalAnalysis.calc_macd(close)
            if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]: val += 0.4
            elif hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2]: val -= 0.4
            
        self.scores["momentum"] = round(np.clip(val, -1.0, 1.0), 3)

    def _analyze_volatility(self):
        df = self.data.get("vix_daily")
        val = 0
        vix = 20.0
        if df is not None and not df.empty:
            vix = df['Close'].iloc[-1]
            if vix < 15: val += 0.2
            elif vix > 20: val -= 0.3
            elif vix > 30: val -= 0.8
        
        self.scores["details"]["vix"] = round(vix, 2)
        self.scores["volatility"] = round(val, 3)

    def _integrate(self):
        total = (self.scores["trend"] * Config.WEIGHT_TREND) + \
                (self.scores["momentum"] * Config.WEIGHT_MOMENTUM) + \
                (self.scores["volatility"] * Config.WEIGHT_VOLATILITY)
        
        # --- Candlestick Pattern Filter ---
        nk_df = self.data.get("nikkei_futures_daily")
        pattern_mult = 1.0
        
        if nk_df is not None and len(nk_df) > 2:
            patterns = TechnicalAnalysis.detect_patterns(nk_df)
            
            # 1. Bullish Patterns
            if patterns.get("bullish_pinbar") or patterns.get("bullish_engulfing"):
                if total > 0: 
                    pattern_mult = 1.3 # Boost Long
                elif total < 0:
                    pattern_mult = 0.1 # Negate Short (Counter-trend risk)

            # 2. Bearish Patterns
            if patterns.get("bearish_pinbar") or patterns.get("bearish_engulfing"):
                if total < 0:
                    pattern_mult = 1.3 # Boost Short
                elif total > 0:
                    pattern_mult = 0.1 # Negate Long (Top reversal risk)
        
        # --- Divergence Filter (Stronger Reversal Warning) ---
        divs = TechnicalAnalysis.detect_divergence(nk_df)
        if divs.get("bearish_divergence") and total > 0:
            pattern_mult = 0.0 # FORCE WAIT (High risk of top)
            # print("   ⚠️ Bearish Divergence -> Long Canceled")
            
        if divs.get("bullish_divergence") and total < 0:
            pattern_mult = 0.0 # FORCE WAIT (High risk of bottom)
            # print("   ⚠️ Bullish Divergence -> Short Canceled")

        total *= pattern_mult
        self.scores["total"] = round(np.clip(total, -1.0, 1.0), 3)
        
        if total > 0.3: self.scores["signal"] = "LONG"
        elif total < -0.3: self.scores["signal"] = "SHORT"
        else: self.scores["signal"] = "WAIT"
        
        self.scores["strength"] = "STRONG" if abs(total) > 0.6 else "MEDIUM"

class GeminiAdvisor:
    """Interfaces with Google Gemini AI."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        if api_key:
            genai.configure(api_key=api_key)

    def consult(self, scores, market_data):
        if not self.api_key:
            return {"approved": True, "reasoning": "No API Key (Simulated)", "direction": scores["signal"]}
            
        prompt = self._create_prompt(scores)
        
        # Multiple model fallback
        for model_name in ['gemini-flash-latest', 'gemini-pro']:
            try:
                model = genai.GenerativeModel(model_name)
                res = model.generate_content(prompt)
                return self._parse_response(res.text, scores["signal"])
            except Exception as e:
                print(f"⚠️  AI Model {model_name} Error: {e}")
                
        return {"approved": True, "reasoning": "AI Unavailable", "direction": scores["signal"]}

    def _create_prompt(self, scores):
        return f"""
        Role: Senior Hedge Fund Manager.
        Task: Review Antigravity Engine Scores and Approve/Reject Trade.
        
        DATA:
        - Trend (Daily): {scores['trend']} ({scores['details'].get('trend_summary')})
        - Momentum (15m): {scores['momentum']} (RSI: {scores['details'].get('rsi')})
        - Volatility (VIX): {scores['volatility']} (VIX: {scores['details'].get('vix')})
        - TOTAL SCORE: {scores['total']}
        - PROPOSED SIGNAL: {scores['signal']}
        
        OUTPUT (JSON Only):
        {{
            "approved": true/false,
            "final_direction": "LONG/SHORT/WAIT",
            "reasoning": "Short succinct analysis."
        }}
        """

    def _parse_response(self, text, default_signal):
        default = {"direction": default_signal, "approved": True, "reasoning": "Parse Failed"}
        if not text: return default
        try:
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return {
                    "direction": data.get("final_direction", default_signal),
                    "approved": data.get("approved", True),
                    "reasoning": data.get("reasoning", "")
                }
        except:
            pass
        return default

class PortfolioManager:
    """Manages Multiple Shadow Portfolios (A/B Testing)."""
    
    def __init__(self):
        self.file_path = Config.DATA_FILE
        self.data = self._load_data()

    def _load_data(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                return json.load(f)
        return {
            "predictions": [], 
            "portfolios": {
                "optimized": {"capital": 100000, "position": None, "trades": []},
                "raptor": {"capital": 100000, "position": None, "trades": []}
            }
        }

    def save(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def update_session(self, current_price_raw: float, low: float, high: float, atr: float):
        """Update ALL portfolios based on their active positions."""
        
        for strat_key, strat_conf in Config.STRATEGIES.items():
            pf = self.data["portfolios"][strat_key]
            
            if not pf.get("position"):
                continue

            pos = pf["position"]
            direction = pos["direction"]
            entry = pos["entry_price"]
            stop = pos["stop"]
            target = pos["target"]
            
            # Checks
            hit_stop = (low <= stop) if direction == "LONG" else (high >= stop)
            hit_target = (high >= target) if direction == "LONG" else (low <= target)
            
            # Time Expiration
            try:
                entry_dt = datetime.strptime(pos["entry_date"], "%Y-%m-%d %H:%M").replace(tzinfo=Config.JST)
                now_dt = datetime.now(Config.JST)
                days_held = (now_dt - entry_dt).days
            except:
                days_held = 99
                
            time_stop = days_held >= Config.MAX_HOLD_DAYS
            
            # Outcome
            exit_price = None
            reason = None
            
            if hit_stop:
                exit_price = stop
                reason = "STOP"
                if hit_target: pass
            elif hit_target:
                exit_price = target
                reason = "TARGET"
            elif time_stop:
                exit_price = round_to_tick(current_price_raw)
                reason = "TIME_STOP"
                
            if exit_price is not None:
                p_diff = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
                gross_pnl = p_diff * Config.CONTRACT_MULTIPLIER * Config.LOTS
                net_pnl = gross_pnl - (Config.COST_PER_TRADE * Config.LOTS)
                
                pf["capital"] += net_pnl
                pf["trades"].append({
                    "entry_date": pos["entry_date"],
                    "exit_date": datetime.now(Config.JST).strftime("%Y-%m-%d %H:%M"),
                    "direction": direction,
                    "entry_price": int(entry),
                    "exit_price": int(exit_price),
                    "pnl_points": int(p_diff),
                    "pnl_yen": int(net_pnl),
                    "close_reason": reason,
                    "days_held": days_held
                })
                pf["position"] = None
                print(f"[{strat_conf['name']}] 🛑 Closed: {reason} PnL: {net_pnl:+}")
            else:
                print(f"[{strat_conf['name']}] 🛌 Hold: Day {days_held}")
            
            self.data["portfolios"][strat_key] = pf

    def open_position(self, prediction, current_price_raw, atr_val):
        """Try to open position for ALL portfolios if not already holding."""
        signal = prediction["direction"]
        if signal not in ["LONG", "SHORT"]: return
        
        entry_price = round_to_tick(current_price_raw)
        safe_atr = atr_val if not pd.isna(atr_val) else 400.0
        
        for strat_key, strat_conf in Config.STRATEGIES.items():
            pf = self.data["portfolios"][strat_key]
            
            if pf.get("position"):
                continue # Already holding in this strategy

            stop_mult = strat_conf["stop_mult"]
            target_mult = strat_conf["target_mult"]
            
            stop_dist = round_to_tick(safe_atr * stop_mult)
            target_dist = round_to_tick(safe_atr * target_mult)
            
            stop_price = entry_price - stop_dist if signal == "LONG" else entry_price + stop_dist
            target_price = entry_price + target_dist if signal == "LONG" else entry_price - target_dist
            
            pf["position"] = {
                "direction": signal,
                "entry_date": datetime.now(Config.JST).strftime("%Y-%m-%d %H:%M"),
                "entry_price": int(entry_price),
                "stop": int(stop_price),
                "target": int(target_price),
                "strategy": strat_key
            }
            self.data["portfolios"][strat_key] = pf
            print(f"[{strat_conf['name']}] 🆕 Entry {signal} @ {entry_price} (Stop:{stop_price} Target:{target_price})")

    def log_prediction(self, scores, prediction, atr):
        # Allow JSON serialization of scores (convert numpy types)
        clean_scores = {
            "trend_score": float(scores["trend"]),
            "momentum_score": float(scores["momentum"]),
            "volatility_score": float(scores["volatility"]),
            "total_score": float(scores["total"]),
            "details": scores["details"]
        }
        
        self.data["predictions"].append({
            "timestamp": datetime.now(Config.JST).strftime("%Y-%m-%d %H:%M"),
            "scores": clean_scores,
            "prediction": prediction,
            "atr": round(atr, 2)
        })

# --- Main App ---

class NikkeiBot:
    def __init__(self):
        self.market = MarketDataManager()
        self.portfolio = PortfolioManager()
        self.advisor = GeminiAdvisor(os.environ.get("GEMINI_API_KEY"))

    def run(self):
        # --- Guard: Skip execution until Feb 6th, 2026 (Launch Prep Phase) ---
        now_jst = datetime.now(Config.JST)
        # Skip if today is 2/6 (or before)
        if now_jst.date() <= datetime(2026, 2, 6).date():
            print(f"🚫 Skipping execution for today ({now_jst.strftime('%Y-%m-%d')}). Launch Prep in progress.")
            return
        # ------------------------------------------------------------

        print("="*60)
        print("🚀 Antigravity Engine v2.1 (A/B Testing Mode)")
        print(f"📅 {datetime.now(Config.JST).strftime('%Y-%m-%d %H:%M JST')}")
        print("="*60)

        # 1. Fetch Data
        print("📥 Fetching Market Data...")
        data = self.market.fetch_all()
        nikkei = data.get("nikkei_futures_daily")
        
        if nikkei is None or nikkei.empty:
            print("❌ Critical Error: No Data")
            sys.exit(1)

        # 2. Analyze
        print("\n🧮 Calculating Scores...")
        engine = AntigravityEngine(data)
        scores = engine.analyze()
        
        print(f"   🌊 Trend:   {scores['trend']:+}")
        print(f"   🚀 Momentum:{scores['momentum']:+}")
        print(f"   ⚠️ Volatility:{scores['volatility']:+}")
        print(f"   💎 TOTAL:    {scores['total']} ({scores['signal']})")
        
        # 3. AI Concurrence
        print("\n🧠 AI Confirmation...")
        prediction = self.advisor.consult(scores, data)
        print(f"   🤖 Verdict: {prediction['direction']} ({prediction['reasoning']})")
        
        # 4. Get Current Metrics
        atr_series = TechnicalAnalysis.calc_atr(nikkei)
        current_atr = atr_series.iloc[-1]
        
        last_row = nikkei.iloc[-1]
        current_price = last_row['Close']
        today_high = last_row['High']
        today_low = last_row['Low']
        
        print(f"   📏 ATR: {current_atr:.0f}")

        # 5. Execute Portfolio Updates
        self.portfolio.update_session(current_price, today_low, today_high, current_atr)
        
        # 6. Record Prediction
        self.portfolio.log_prediction(scores, prediction, current_atr)
        
        # 7. Open New Positions (A/B Test) - RiskGate Disabled
        gap_wait = False
        # prev_close = nikkei.iloc[-2]['Close']
        # gap_rate = abs(current_price - prev_close) / prev_close
        
        # if gap_rate >= Config.GAP_THRESHOLD:
        #     print(f"⚠️ RiskGate Triggered: Gap {gap_rate*100:.2f}% >= {Config.GAP_THRESHOLD*100:.1f}%")
        #     print("   ⛔ Entry Canceled to avoid gap risk.")
        #     gap_wait = True
            
        if prediction['approved'] and not gap_wait:
            self.portfolio.open_position(prediction, current_price, current_atr)
        
        # 8. Save State
        self.portfolio.save()
        
        # 9. Report Stats (A/B)
        print("\n📈 Strategy Performance:")
        portfolios = self.portfolio.data["portfolios"]
        
        for key, conf in Config.STRATEGIES.items():
            pf = portfolios[key]
            trades = pf["trades"]
            capital = pf["capital"]
            
            wins = len([t for t in trades if t['pnl_yen'] > 0])
            total_trades = len(trades)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
            total_pnl = sum(t['pnl_yen'] for t in trades)
            
            pos_str = f"{pf['position']['direction']}@{pf['position']['entry_price']}" if pf['position'] else "FLAT"
            
            print(f" [{conf['name']}]")
            print(f"   💰 Cap: ¥{capital:,.0f} ({total_pnl:+})")
            print(f"   📊 Win: {win_rate:.1f}% ({wins}/{total_trades})")
            print(f"   📍 Pos: {pos_str}")
            print("-" * 30)
            
        print("✅ Done.")

if __name__ == "__main__":
    bot = NikkeiBot()
    bot.run()
