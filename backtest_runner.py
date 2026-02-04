import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Import Core Logic from src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.nikkei_bot import Config, TechnicalAnalysis, AntigravityEngine, round_to_tick

# --- Backtest Configuration ---
INITIAL_CAPITAL = 100000
START_DATE = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
MODE_OVERNIGHT = True

# --- Simulation Engine ---

class BacktestEngine(AntigravityEngine):
    """
    Extends production engine to support daily-data-only backtesting.
    Overrides detailed intraday checks with daily proxies.
    """
    def _analyze_momentum(self):
        # Override: Use Daily data instead of 15m for backtest proxy
        df = self.data.get("nikkei_futures_daily")
        val = 0
        if df is not None and len(df) > 30:
            close = df['Close']
            
            # RSI (Daily proxy)
            rsi = TechnicalAnalysis.calc_rsi(close).iloc[-1]
            self.scores["details"]["rsi"] = round(rsi, 1) if not np.isnan(rsi) else 50
            
            if rsi > 70: val -= 0.3
            elif rsi < 30: val += 0.3
            elif rsi > 50: val += 0.2
            else: val -= 0.2
            
            # MACD (Daily)
            _, _, hist = TechnicalAnalysis.calc_macd(close)
            if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]: val += 0.4
            elif hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2]: val -= 0.4
            
        self.scores["momentum"] = round(np.clip(val, -1.0, 1.0), 3)

def run_backtest():
    print(f"📥 Fetching 3 years of data (Start: {START_DATE})...")
    
    # 1. Fetch Long History
    # We use yfinance directly here as MarketDataManager fetches fixed periods (1y)
    nikkei = yf.download(Config.TICKERS["nikkei_index"], start=START_DATE, progress=False)
    vix = yf.download(Config.TICKERS["vix"], start=START_DATE, progress=False)
    
    # Flatten MultiIndex if needed
    if isinstance(nikkei.columns, pd.MultiIndex): nikkei.columns = nikkei.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

    # Pre-calc ATR (we need it for the loop)
    # We'll use TechnicalAnalysis lib but apply it to the whole df
    nikkei['ATR'] = TechnicalAnalysis.calc_atr(nikkei)
    
    # Reindex VIX
    vix = vix['Close'].reindex(nikkei.index).fillna(20.0)
    
    print("🚀 Running Antigravity Simulation...")
    print(f"   Strategy: Swing (Max {Config.MAX_HOLD_DAYS} Days)")
    print(f"   Risk: Stop {Config.RISK_STOP_ATR_MULT}x / Target {Config.RISK_TARGET_ATR_MULT}x")
    
    capital = INITIAL_CAPITAL
    trades = []
    position = None 
    
    # Loop
    # Start from index 200 to have enough data for EMA200
    for i in range(200, len(nikkei)-1):
        
        # Prepare Data Slice (Simulate "Past")
        # We need a slice up to 'i' for indicators.
        # Ideally we pass the full series and tell the engine "current index = i",
        # but our Engine takes a DataFrame and looks at .iloc[-1].
        # So we pass a slice up to i.
        
        current_date_idx = nikkei.index[i]
        
        # Slicing is heavy, but necessary for strict simulation using the Class
        # Optimization: Pass a window
        window = nikkei.iloc[i-250 : i+1] 
        vix_window = vix.iloc[i-50 : i+1].to_frame(name='Close')
        
        market_data_slice = {
            "nikkei_futures_daily": window,
            "vix_daily": vix_window,
            # "nikkei_15m": None # BacktestEngine handles missing 15m
        }
        
        # --- 1. Position Management (Morning of Day i) ---
        # Note: We are at Close of Day 'i'.
        # In reality, we manage positions based on price movement of Day 'i'.
        
        today_row = nikkei.iloc[i]
        today_open = float(today_row['Open'])
        today_high = float(today_row['High'])
        today_low = float(today_row['Low'])
        today_close = float(today_row['Close'])
        
        if position:
            p_type = position['type']
            stop = position['stop']
            target = position['target']
            
            hit_stop = False
            hit_target = False
            exit_price = today_close
            
            # Check Gaps & OHLC
            if p_type == "LONG":
                if today_open <= stop: hit_stop = True; exit_price = today_open
                elif today_open >= target: hit_target = True; exit_price = today_open
                elif today_low <= stop: hit_stop = True; exit_price = stop
                elif today_high >= target: hit_target = True; exit_price = target
            else: # SHORT
                if today_open >= stop: hit_stop = True; exit_price = today_open
                elif today_open <= target: hit_target = True; exit_price = today_open
                elif today_high >= stop: hit_stop = True; exit_price = stop
                elif today_low <= target: hit_target = True; exit_price = target
            
            position['days'] += 1
            time_stop = position['days'] >= Config.MAX_HOLD_DAYS
            
            if hit_stop or hit_target or time_stop:
                if time_stop and not hit_stop and not hit_target:
                    exit_price = round_to_tick(today_close)
                    
                gross = (exit_price - position['entry']) if p_type == "LONG" else (position['entry'] - exit_price)
                gross *= Config.CONTRACT_MULTIPLIER
                net = gross - Config.COST_PER_TRADE
                capital += net
                trades.append(net)
                position = None
                continue # Position closed, wait for next signal (tomorrow)

        # --- 2. Signal Generation (Close of Day i) ---
        # If we have a position, we skip entry (Standard Swing Rule)
        if position: continue
        
        engine = BacktestEngine(market_data_slice)
        scores = engine.analyze()
        signal = scores['signal'] # LONG, SHORT, WAIT
        
        if signal == "WAIT": continue
        
        # --- 3. Entry Setup (Market Open of Day i+1) ---
        tomorrow_row = nikkei.iloc[i+1] # We can peek tomorrow O for entry
        entry_price = round_to_tick(float(tomorrow_row['Open']))
        atr = float(window['ATR'].iloc[-1])
        if pd.isna(atr): atr = 400.0
        
        stop_dist = round_to_tick(atr * Config.RISK_STOP_ATR_MULT)
        target_dist = round_to_tick(atr * Config.RISK_TARGET_ATR_MULT)
        
        position = {
            'type': signal,
            'entry': entry_price,
            'stop': entry_price - stop_dist if signal == "LONG" else entry_price + stop_dist,
            'target': entry_price + target_dist if signal == "LONG" else entry_price - target_dist,
            'days': 0
        }

    # Result Calculation
    total_trades = len(trades)
    wins = len([t for t in trades if t > 0])
    win_rate = (wins/total_trades*100) if total_trades else 0
    ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    print("\n" + "="*40)
    print(f"🏁 BACKTEST RESULT (3 Years)")
    print("="*40)
    print(f"💰 Final Capital:  ¥{capital:,.0f}")
    print(f"📈 Total Return:   {ret:+.2f}%")
    print(f"📊 Win Rate:       {win_rate:.1f}% ({wins}/{total_trades})")
    print("="*40)

if __name__ == "__main__":
    run_backtest()
