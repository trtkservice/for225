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
        if position: continue
        
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

    today = datetime.now()
    start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
    months = (today - start_dt).days / 30.417
    
    print(f"🔎 Comparative Search: SWING vs DAY TRADE (RiskGate: {Config.GAP_THRESHOLD*100:.1f}% - DISABLED)")
    print(f"   Period: {months:.1f} months")
    print("="*105)
    print(f"STOP |TGT  || SWING RET  (Win%) [Avg/Mo] || DAY RET    (Win%) [Avg/Mo]")
    print("-" * 105)
    
    for s, t in itertools.product(STOP_RANGE, TARGET_RANGE):
        if t <= s: continue 
        
        # Run SWING
        cap_s, trds_s = run_simulation(nikkei, vix, s, t, mode="SWING")
        ret_s = (cap_s - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        win_s = (len([x for x in trds_s if x > 0])/len(trds_s)*100) if trds_s else 0
        pnl_s = cap_s - INITIAL_CAPITAL
        avg_s = pnl_s / months
        
        # Run DAY
        cap_d, trds_d = run_simulation(nikkei, vix, s, t, mode="DAY")
        ret_d = (cap_d - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        win_d = (len([x for x in trds_d if x > 0])/len(trds_d)*100) if trds_d else 0
        pnl_d = cap_d - INITIAL_CAPITAL
        avg_d = pnl_d / months
        
        # Highlight winner
        winner = "SWING" if ret_s > ret_d else "DAY  "
        if ret_s < 0 and ret_d < 0: winner = "LOSE"
        
        # Print results
        print(f"  {s:<4}|{t:<4}|| {ret_s:>+8.1f}% ({win_s:>4.1f}%) [{avg_s:>+6.0f}] || {ret_d:>+8.1f}% ({win_d:>4.1f}%) [{avg_d:>+6.0f}] {winner}")
    print("="*105)

if __name__ == "__main__":
    run_grid_search()
