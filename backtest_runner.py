
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import itertools
import sys
import os

# Put src in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from nikkei_bot import Config, TechnicalAnalysis, LiLFlexxEngine, round_to_tick

# --- Backtest Configuration ---
INITIAL_CAPITAL = 100000
START_DATE = (datetime.now() - timedelta(days=20*365)).strftime('%Y-%m-%d')
BACKTEST_LOTS = 1

# Rakuten Securities Cost Simulation
SPREAD = 5.0 # JPY (Entry + Exit Slippage)
COST_PER_TRADE = 75 # JPY (Commission + Hidden Cost)

class BacktestEngine(LiLFlexxEngine):
    """
    Simpler engine for backtesting.
    """
    def __init__(self, data):
        super().__init__(data)

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
    date_index = nikkei.index.tolist()
    atr_records = nikkei['ATR'].tolist()
    
    # Start loop (skip enough data for indicators)
    for i in range(200, len(daily_records)-1):
        today = daily_records[i]
        current_date = date_index[i]
        
        open_p, high_p, low_p, close_p = today['Open'], today['High'], today['Low'], today['Close']
        
        # 1. Manage Position
        if position:
            p_type = position['type']
            stop = position['stop']
            target = position['target']
            
            # Check Exit
            hit_stop = False
            hit_target = False
            exit_price = None
            
            if p_type == "LONG":
                if low_p <= stop: exit_price = stop; hit_stop = True
                elif high_p >= target: exit_price = target; hit_target = True
                # Strict Stop Check (Spread)
                # Low is Bid. Bid <= Stop -> HIT
                
            elif p_type == "SHORT":
                if high_p >= stop: exit_price = stop; hit_stop = True
                elif low_p <= target: exit_price = target; hit_target = True
                # Strict Check: Ask hits stop. Ask = High + Spread
                elif high_p + SPREAD >= stop: exit_price = stop; hit_stop = True
                
            # Forced Exit Logic
            is_timestop = False
            if mode == "DAY":
                # For DayTrade, if not hit stop/target, exit at Close
                if not hit_stop and not hit_target:
                    exit_price = close_p
                    is_timestop = True
            elif mode == "DOTEN":
                # DOTEN: Only exit if Signal Reverses (handled below) or Hit Stop
                position['days'] += 1
                if position['days'] >= 20: 
                    is_timestop = True
            else:
                # SWING
                position['days'] += 1
                if position['days'] >= Config.MAX_HOLD_DAYS:
                    is_timestop = True

            if hit_stop or hit_target or is_timestop:
                if not exit_price: exit_price = close_p 
                
                # Calc PnL
                diff = (exit_price - position['entry']) if p_type == "LONG" else (position['entry'] - exit_price)
                bn = (diff * Config.CONTRACT_MULTIPLIER * BACKTEST_LOTS) - (COST_PER_TRADE * BACKTEST_LOTS)
                capital += bn
                trades.append({'date': current_date, 'pnl': bn})
                position = None
                continue

        # 2. Entry Signal
        window = nikkei.iloc[i-50:i+1].copy()
        vix_window = vix.iloc[i-50:i+1].copy()
        
        engine = BacktestEngine({
            "nikkei_futures_daily": window,
            "vix_daily": vix_window
        })
        scores = engine.analyze()
        signal = scores['signal']
        
        if position: continue
        if signal == "WAIT": continue
        
        # 3. Enter Next Open
        next_day = daily_records[i+1]
        entry_price = round_to_tick(next_day['Open'])
            
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
    pnl_values = [t['pnl'] for t in trades]
    total_profit = sum([x for x in pnl_values if x > 0])
    total_loss = abs(sum([x for x in pnl_values if x < 0]))
    pf = (total_profit / total_loss) if total_loss > 0 else 0
    
    # Max DD
    peak = INITIAL_CAPITAL
    max_dd = 0
    running_cap = INITIAL_CAPITAL
    for t in pnl_values:
        running_cap += t
        if running_cap > peak: peak = running_cap
        dd = (peak - running_cap) / peak * 100
        if dd > max_dd: max_dd = dd
        
    stats = {
        "capital": capital,
        "trades": len(trades),
        "win_rate": (len([x for x in pnl_values if x > 0]) / len(pnl_values) * 100) if pnl_values else 0,
        "pf": pf,
        "max_dd": max_dd,
        "return": (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "details": trades
    }
    return stats

def run_top10_analysis():
    print(f"📥 Fetching Data (20 Years)...")
    nikkei = yf.download(Config.TICKERS["nikkei_index"], start=START_DATE, progress=False)
    vix = yf.download(Config.TICKERS["vix"], start=START_DATE, progress=False)
    
    if isinstance(nikkei.columns, pd.MultiIndex): nikkei.columns = nikkei.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

    # Pre-calc ATR
    nikkei['ATR'] = TechnicalAnalysis.calc_atr(nikkei)
    vix = vix['Close'].reindex(nikkei.index).fillna(20.0).to_frame(name='Close')
    
    # Top 10 Strategies
    TOP_10 = [
        (0.4, 3.0), (0.4, 2.5), (0.5, 3.0), (0.5, 2.5), (0.4, 2.0),
        (0.6, 3.0), (0.4, 1.5), (0.4, 1.2), (0.5, 2.0), (0.6, 2.5)
    ]
    
    print(f"🔎 Detailed Analysis: Top 10 Strategies (2006-2026)")
    print(f"   Spread: {SPREAD} | Cost: {COST_PER_TRADE} | Lots: {BACKTEST_LOTS}")
    print("="*80)
    
    for s, t in TOP_10:
        res = run_simulation(nikkei, vix, s, t, mode="DAY")
        print(f"Strategy: Stop {s} / Target {t} | Total Return: {res['return']:+.1f}% | PF: {res['pf']:.2f} | MaxDD: {res['max_dd']:.1f}%")
        
        # Yearly PnL Breakdown (Only for the #1 Strategy)
        if s == 0.4 and t == 3.0:
            print("-" * 60)
            print("   📅 Yearly Breakdown (Profit/Loss in JPY)")
            
            yearly_pnl = {}
            for t_log in res['details']:
                year = t_log['date'].year
                yearly_pnl[year] = yearly_pnl.get(year, 0) + t_log['pnl']
            
            cumulative = 0
            for year in sorted(yearly_pnl.keys()):
                pnl = yearly_pnl[year]
                cumulative += pnl
                print(f"   {year}: {pnl:>+10,.0f} JPY  (Cum: {cumulative:>+10,.0f})")
            print("-" * 60)
    
    print("="*80)

if __name__ == "__main__":
    run_top10_analysis()
