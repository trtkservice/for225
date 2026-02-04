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

# --- Optimization Settings ---
STOP_RANGE = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
TARGET_RANGE = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

def run_simulation(nikkei, vix, stop_mult, target_mult):
    """Single Backtest Run for given parameters."""
    capital = INITIAL_CAPITAL
    trades = []
    position = None 
    
    # Temporarily Patch Config
    Config.RISK_STOP_ATR_MULT = stop_mult
    Config.RISK_TARGET_ATR_MULT = target_mult
    
    # Loop (Optimized for speed)
    # Note: We assume 'nikkei' df already has 'ATR' column
    
    daily_records = nikkei.to_dict('records')
    atr_records = nikkei['ATR'].tolist()
    
    # Start mainly after EMA200 (index 200)
    for i in range(200, len(daily_records)-1):
        today = daily_records[i]
        
        # 1. Manage Position
        if position:
            p_type = position['type']
            stop = position['stop']
            target = position['target']
            
            # Simulated OHLC Check
            open_p, high_p, low_p, close_p = today['Open'], today['High'], today['Low'], today['Close']
            
            exit_price = None
            reason = ""
            
            # Logic: Check Open gap -> then High/Low -> then Time
            hit_stop = False
            hit_target = False
            
            if p_type == "LONG":
                if open_p <= stop: exit_price = open_p; hit_stop = True
                elif open_p >= target: exit_price = open_p; hit_target = True
                elif low_p <= stop: exit_price = stop; hit_stop = True
                elif high_p >= target: exit_price = target; hit_target = True
            else: # SHORT
                if open_p >= stop: exit_price = open_p; hit_stop = True
                elif open_p <= target: exit_price = open_p; hit_target = True
                elif high_p >= stop: exit_price = stop; hit_stop = True
                elif low_p <= target: exit_price = target; hit_target = True
            
            position['days'] += 1
            is_timestop = position['days'] >= Config.MAX_HOLD_DAYS
            
            if hit_stop or hit_target or is_timestop:
                if not exit_price: exit_price = close_p # Time stop close
                
                # Calc PnL
                # Compound: Lots logic was decided at entry, stored in position
                lots = position.get('lots', 1)
                
                diff = (exit_price - position['entry']) if p_type == "LONG" else (position['entry'] - exit_price)
                gross = (diff * Config.CONTRACT_MULTIPLIER * lots)
                net = gross - (Config.COST_PER_TRADE * lots)
                
                capital += net
                trades.append(net)
                position = None
                continue

        # 2. Entry Signal (Only if no position)
        if position: continue
        
        # Create slice for analysis (Costly but needed for strict class usage)
        # To speed up, we might cheat and pre-calculate signals, but let's be strict first.
        # Check simple basic rule to avoid creating Class object every time?
        # No, let's trust Python speed.
        
        # Slice: Need enough history for EMA200
        start_idx = i - 250
        if start_idx < 0: start_idx = 0
        window = pd.DataFrame(daily_records[start_idx : i+1])
        vix_window = pd.DataFrame(vix.iloc[start_idx : i+1]) # VIX is simple series
        
        engine = BacktestEngine({
            "nikkei_futures_daily": window,
            "vix_daily": vix_window
        })
        scores = engine.analyze()
        signal = scores['signal']
        
        if signal == "WAIT": continue
        
        # 3. Enter Next Open
        next_day = daily_records[i+1]
        entry_price = round_to_tick(next_day['Open'])
        atr = atr_records[i]
        if pd.isna(atr): atr = 400.0
        
        s_dist = round_to_tick(atr * stop_mult)
        t_dist = round_to_tick(atr * target_mult)
        
        # Compound Sizing
        lots = int(capital / 100000)
        if lots < 1: lots = 1
        
        position = {
            'type': signal,
            'entry': entry_price,
            'stop': entry_price - s_dist if signal == "LONG" else entry_price + s_dist,
            'target': entry_price + t_dist if signal == "LONG" else entry_price - t_dist,
            'days': 0,
            'lots': lots
        }
        
    return capital, trades


def run_grid_search():
    print(f"📥 Fetching Data...")
    nikkei = yf.download(Config.TICKERS["nikkei_index"], start=START_DATE, progress=False)
    vix = yf.download(Config.TICKERS["vix"], start=START_DATE, progress=False)
    
    if isinstance(nikkei.columns, pd.MultiIndex): nikkei.columns = nikkei.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

    # Pre-calc ATR
    nikkei['ATR'] = TechnicalAnalysis.calc_atr(nikkei)
    vix = vix['Close'].reindex(nikkei.index).fillna(20.0).to_frame(name='Close')

    print(f"🔎 Starting Grid Search (Stop: {STOP_RANGE}, Target: {TARGET_RANGE})")
    print(f"   Total Combinations: {len(STOP_RANGE) * len(TARGET_RANGE)}")
    print("="*60)
    print(f"{'STOP':<6} | {'TARGET':<6} | {'RETURN':<10} | {'WIN %':<8} | {'TRADES':<6}")
    print("-" * 60)
    
    results = []
    
    import itertools
    for s, t in itertools.product(STOP_RANGE, TARGET_RANGE):
        if t <= s: continue # Skip if Target <= Stop (bad R:R)
        
        cap, trds = run_simulation(nikkei, vix, s, t)
        
        ret = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        wins = len([x for x in trds if x > 0])
        total = len(trds)
        win_rate = (wins/total*100) if total else 0
        
        # Print progress
        # print(f" {s:<6} | {t:<6} | {ret:>+9.1f}% | {win_rate:>6.1f}% | {total:<6}")
        
        results.append({
            "stop": s, "target": t, "return": ret, "win_rate": win_rate, "trades": total
        })

    # Sort by Return
    print("-" * 60)
    print("🏆 TOP 10 SETTINGS 🏆")
    print("-" * 60)
    results.sort(key=lambda x: x['return'], reverse=True)
    
    for r in results[:10]:
         print(f" Stop:{r['stop']} / Tgt:{r['target']}  => 💰 {r['return']:>+6.1f}%  (Win: {r['win_rate']:.1f}% / {r['trades']} trds)")
         
    # Check current setting rank
    curr_s = 0.6
    curr_t = 1.2
    found = next((r for r in results if r['stop']==curr_s and r['target']==curr_t), None)
    if found:
        print("-" * 60)
        print(f"👉 Current Setting ({curr_s}/{curr_t}): {found['return']:+.1f}% (Rank: {results.index(found)+1}/{len(results)})")

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
    run_grid_search()
