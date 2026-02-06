import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import itertools
from datetime import datetime, timedelta

# Import Core Logic from src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.nikkei_bot import Config, TechnicalAnalysis, AntigravityEngine, round_to_tick

# --- Backtest Configuration ---
INITIAL_CAPITAL = 100000
START_DATE = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
BACKTEST_LOTS = 1

# Rakuten Securities Cost Simulation
SPREAD = 5.0                # Spread/Slippage (price units)
COST_PER_TRADE = 75         # Total Cost per trade (Spread + Commission)

# --- Optimization Settings ---
STOP_RANGE = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
TARGET_RANGE = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

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

def run_simulation(nikkei, vix, stop_mult, target_mult, mode="SWING"):
    """
    mode: "SWING" (Overnight) or "DAY" (Exit at Close)
    """
    capital = INITIAL_CAPITAL
    trades = []
    position = None 
    
    # Temporarily Patch Config
    Config.RISK_STOP_ATR_MULT = stop_mult
    Config.RISK_TARGET_ATR_MULT = target_mult
    
    daily_records = nikkei.to_dict('records')
    atr_records = nikkei['ATR'].tolist()
    
    # Start loop (skip enough data for indicators)
    for i in range(200, len(daily_records)-1):
        today = daily_records[i]
        
        # 1. Manage Position
        if position:
            p_type = position['type']
            stop = position['stop']
            target = position['target']
            
            # OHLC
            open_p, high_p, low_p, close_p = today['Open'], today['High'], today['Low'], today['Close']
            
            # Check Stop / Target
            hit_stop = False
            hit_target = False
            exit_price = None
            
            if position['type'] == 'LONG':
                # Conservative Open check: Did we open below stop?
                if open_p <= stop: exit_price = open_p; hit_stop = True
                elif open_p >= target: exit_price = open_p; hit_target = True
                
                # Strict Stop Check (Accounting for Spread/Ask price)
                # We get stopped out if the BID price hits stop. 
                # Assuming Low is Bid. But in panic, slippage occurs.
                elif low_p - SPREAD <= stop: exit_price = stop; hit_stop = True
                elif high_p >= target: exit_price = target; hit_target = True
            else: # SHORT
                if open_p >= stop: exit_price = open_p; hit_stop = True
                elif open_p <= target: exit_price = open_p; hit_target = True
                
                # Strict Stop Check
                # Stopped if ASK price hits stop. High is Bid, Ask is High + Spread.
                elif high_p + SPREAD >= stop: exit_price = stop; hit_stop = True
                elif low_p <= target: exit_price = target; hit_target = True
                
            # --- Trailing Stop Logic Removed (Reverted to Fixed Logic in previous step) ---
            
            # Forced Exit Logic
            is_timestop = False
            if mode == "DAY":
                # For DayTrade, if not hit stop/target, exit at Close
                if not hit_stop and not hit_target:
                    exit_price = close_p
                    is_timestop = True
            elif mode == "DOTEN":
                # DOTEN: Only exit if Signal Reverses (handled at start of loop) or Hit Stop
                # Here we just handle time limit (hold days) just in case
                position['days'] += 1
                if position['days'] >= 20: # Long term hold limit
                    is_timestop = True
            else:
                # SWING
                position['days'] += 1
                if position['days'] >= Config.MAX_HOLD_DAYS:
                    is_timestop = True

            if hit_stop or hit_target or is_timestop:
                if not exit_price: exit_price = close_p 
                
                # Calc PnL (Fixed Lot - Single Interest)
                diff = (exit_price - position['entry']) if p_type == "LONG" else (position['entry'] - exit_price)
                bn = (diff * Config.CONTRACT_MULTIPLIER * BACKTEST_LOTS) - (COST_PER_TRADE * BACKTEST_LOTS)
                capital += bn
                trades.append(bn)
                position = None
                continue

        # 2. Entry Signal
        # In DOTEN mode, if we have a position, we check if signal reverses
        
        # Analyze
        start_idx = i - 250
        if start_idx < 0: start_idx = 0
        window = pd.DataFrame(daily_records[start_idx : i+1])
        vix_window = pd.DataFrame(vix.iloc[start_idx : i+1])
        
        engine = BacktestEngine({
            "nikkei_futures_daily": window,
            "vix_daily": vix_window
        })
        scores = engine.analyze()
        signal = scores['signal']
        
        # DOTEN Reversal Logic
        if mode == "DOTEN" and position:
            # If Signal opposes Position -> Reverse
            # (Note: We are at end of day 'i', decision for day 'i+1')
            reverse = False
            if position['type'] == 'LONG' and signal == 'SHORT': reverse = True
            elif position['type'] == 'SHORT' and signal == 'LONG': reverse = True
            
            if reverse:
                # Close current position at Next Open
                next_open = round_to_tick(daily_records[i+1]['Open'])
                diff = (next_open - position['entry']) if position['type'] == 'LONG' else (position['entry'] - next_open)
                bn = (diff * Config.CONTRACT_MULTIPLIER * BACKTEST_LOTS) - (COST_PER_TRADE * BACKTEST_LOTS)
                capital += bn
                trades.append(bn)
                position = None # Closed, will allow new entry below
                
                # BUT, wait, "signal" is for next entry. 
                # If we close here, we fall through to "if position: continue" check?
                # No, we set position = None, so it proceeds to enter new position below.
                
                # Correction: If we reverse, we need to enter IMMEDIATELY at Next Open.
                # So we cannot just set position=None and fall through, because fall-through logic is "normal entry".
                # We should handle re-entry here or use a flag.
                # Simplest: Close here, and let the normal entry logic below pick it up 
                # because we set position = None.
                pass

        if position: continue
        
        if signal == "WAIT": continue
        
        # 3. Enter Next Open
        next_day = daily_records[i+1]
        entry_price = round_to_tick(next_day['Open'])
        
        # RiskGate Check (Gap Filter) - DISABLED
        # prev_close = daily_records[i]['Close']
        # gap_rate = abs(entry_price - prev_close) / prev_close
        # if gap_rate >= Config.GAP_THRESHOLD:
        #     continue
            
        atr = atr_records[i]
        if pd.isna(atr): atr = 400.0
        
        s_dist = round_to_tick(atr * stop_mult)
        t_dist = round_to_tick(atr * target_mult)
        
        position = {
            'type': signal,
            'entry': entry_price,
            'stop': entry_price - s_dist if signal == "LONG" else entry_price + s_dist,
            'target': entry_price + t_dist if signal == "LONG" else entry_price - t_dist,
            'days': 0
        }
        
    # --- Calc Stats ---
    total_profit = sum([x for x in trades if x > 0])
    total_loss = abs(sum([x for x in trades if x < 0]))
    pf = (total_profit / total_loss) if total_loss > 0 else 0
    
    # Max DD
    peak = INITIAL_CAPITAL
    max_dd = 0
    running_cap = INITIAL_CAPITAL
    for t in trades:
        running_cap += t
        if running_cap > peak: peak = running_cap
        dd = (peak - running_cap) / peak * 100
        if dd > max_dd: max_dd = dd
        
    stats = {
        "capital": capital,
        "trades": len(trades),
        "win_rate": (len([x for x in trades if x > 0]) / len(trades) * 100) if trades else 0,
        "pf": pf,
        "max_dd": max_dd,
        "return": (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    }
    return stats


def run_grid_search():
    print(f"📥 Fetching Data...")
    nikkei = yf.download(Config.TICKERS["nikkei_index"], start=START_DATE, progress=False)
    vix = yf.download(Config.TICKERS["vix"], start=START_DATE, progress=False)
    
    if isinstance(nikkei.columns, pd.MultiIndex): nikkei.columns = nikkei.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

    # Pre-calc ATR
    nikkei['ATR'] = TechnicalAnalysis.calc_atr(nikkei)
    vix = vix['Close'].reindex(nikkei.index).fillna(20.0).to_frame(name='Close')

    today = datetime.now()
    start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
    months = (today - start_dt).days / 30.417
    
    print(f"🔎 Comparative Search: DAY vs DOTEN vs SWING")
    print(f"   Period: {months:.1f} months | Spread: {SPREAD} | Cost: {COST_PER_TRADE}")
    print("="*140)
    print(f"STOP |TGT  || DAY [Ret/PF/DD]       || DOTEN [Ret/PF/DD]")
    print("-" * 140)
    
    for s, t in itertools.product(STOP_RANGE, TARGET_RANGE):
        if t <= s: continue 
        
        # Run DAY
        res_d = run_simulation(nikkei, vix, s, t, mode="DAY")
        
        # Run DOTEN
        res_o = run_simulation(nikkei, vix, s, t, mode="DOTEN")

        # Winner
        winner = "DAY  " if res_d["return"] >= res_o["return"] else "DOTEN"
        
        # Print
        row_d = f"{res_d['return']:>+6.0f}% / {res_d['pf']:4.2f} / {res_d['max_dd']:4.1f}%"
        row_o = f"{res_o['return']:>+6.0f}% / {res_o['pf']:4.2f} / {res_o['max_dd']:4.1f}%"
        
        print(f"  {s:<4}|{t:<4}|| {row_d:<24} || {row_o:<24} {winner}")
    print("="*140)

if __name__ == "__main__":
    run_grid_search()
