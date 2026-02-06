
import pandas as pd
import numpy as np
import glob
import os
import sys
from datetime import datetime, time, timedelta
import itertools
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Config
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from nikkei_bot import Config, round_to_tick

# Data Directory
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Simulation Config
INITIAL_CAPITAL = 100000 # 100k start
BACKTEST_LOTS = 2        # 2 Micro Lots
MULTIPLIER = 10          # Micro = 10x
COST_PER_TRADE = 60      # Rakuten Micro ~11JPY x 2 x 2 sides + Slippage

# --- Raptor Best Settings ---
BEST_G = 0.0025
BEST_N = 48
BEST_R = 1.8

# Search Range for Risk Management (Focus on Best Setting)
STOP_RANGE = [1.0]
TARGET_RANGE = [2.0]

def load_and_merge_data():
    """Load all N225minif_*.xlsx files and merge them."""
    pattern = os.path.join(DATA_DIR, "N225minif_*.xlsx")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("❌ No excel files found matching 'N225minif_*.xlsx'")
        sys.exit(1)
        
    print(f"📥 Loading {len(files)} Excel files... (This may take a minute)")
    
    df_list = []
    for f in files:
        print(f"   Reading {os.path.basename(f)}...")
        try:
            temp = pd.read_excel(f)
            df_list.append(temp)
        except Exception as e:
            print(f"   ⚠️ Failed to read {f}: {e}")

    full_df = pd.concat(df_list, ignore_index=True)
    
    # Rename map
    rename_map = {
        '日付': 'Date', 'Date': 'Date', 'date': 'Date',
        '時間': 'Time', '時刻': 'Time', 'Time': 'Time', 'time': 'Time',
        '始値': 'Open', 'Open': 'Open', 'open': 'Open',
        '高値': 'High', 'High': 'High', 'high': 'High',
        '安値': 'Low', 'Low': 'Low', 'low': 'Low',
        '終値': 'Close', 'Close': 'Close', 'close': 'Close'
    }
    full_df.rename(columns=rename_map, inplace=True)
    
    # Combine Date+Time to Datetime Index
    def parse_datetime(row):
        d = row['Date']
        t = row['Time']
        if isinstance(d, str): d = datetime.strptime(d, '%Y/%m/%d').date()
        if isinstance(d, datetime): d = d.date()
        if isinstance(t, str): t = datetime.strptime(t, '%H:%M').time()
        return datetime.combine(d, t)

    print("   Processing timestamps...")
    try:
        full_df['Datetime'] = pd.to_datetime(full_df['Date'].astype(str) + ' ' + full_df['Time'].astype(str))
    except:
        full_df['Datetime'] = full_df.apply(parse_datetime, axis=1)

    full_df.set_index('Datetime', inplace=True)
    full_df.sort_index(inplace=True)
    full_df = full_df[~full_df.index.duplicated(keep='first')]
    
    # Keep only OHLC
    df = full_df[['Open', 'High', 'Low', 'Close']].astype(float)
    
    print(f"✅ Loaded {len(df)} 1-minute bars ({df.index[0]} to {df.index[-1]})")
    return df

def create_session_data(df_1m):
    """
    Create Session Data (DAY / NIGHT) and 15m Data.
    """
    print("🔄 Creating Session & 15m Data...")
    
    # 1. 15m Data
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    
    return df_15m

def calculate_slope(series):
    """Calculate slope of regression line for the series."""
    y = series.values
    x = np.arange(len(y))
    if len(x) < 2: return 0
    # Linear regression: y = ax + b
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a

def run_raptor_simulation_risk(df_1m, df_15m, g_cut, n_period, stop_mult, target_mult, df_daily):
    capital = INITIAL_CAPITAL
    trades = []
    win_count = 0
    trade_count = 0
    
    unique_dates = sorted(list(set(df_1m.index.date)))
    months = len(set([d.strftime('%Y-%m') for d in unique_dates]))
    
    for i in range(1, len(unique_dates)):
        curr_date = unique_dates[i]
        prev_date = unique_dates[i-1]
        
        # --- RAPTOR LOGIC START ---
        
        day_start = datetime.combine(curr_date, time(8, 45))
        day_end = datetime.combine(curr_date, time(15, 15))
        
        day_data = df_1m.loc[day_start:day_end]
        if day_data.empty: continue
        entry_price = day_data.iloc[0]['Open']
        
        night_start = datetime.combine(prev_date, time(16, 30))
        night_end = datetime.combine(curr_date, time(6, 0))
        night_data = df_1m.loc[night_start:night_end]
        
        # Allow missing night data (Monday gap) but default to Skip if strict
        if night_data.empty: continue
            
        prev_close = night_data.iloc[-1]['Close']
        prev_open = night_data.iloc[0]['Open']
        
        # 1. Gap Check (RiskGate)
        gap_rate = (entry_price - prev_close) / prev_close
        if abs(gap_rate) >= g_cut: continue
        
        # 2. B: Night Trend
        score_b = 1 if prev_close > prev_open else -1
        
        # 3. C: Momentum Slope
        recent_15m = df_15m.loc[:day_start].iloc[-(n_period+1):-1]
        if len(recent_15m) < n_period: continue
        slope = calculate_slope(recent_15m['Close'])
        score_c = 1 if slope > 0 else -1
        
        total = score_b + score_c
        action = "NO-TRADE"
        if total >= 2: action = "BUY"
        elif total <= -2: action = "SELL"
        
        if action == "NO-TRADE": continue
        
        # --- RISK EXECUTION ---
        atr = 300 # Default
        try:
            atr_s = df_daily.loc[str(prev_date)]['ATR']
            atr = atr_s if not pd.isna(atr_s) else 300
        except: pass
        
        s_dist = round_to_tick(atr * stop_mult)
        t_dist = round_to_tick(atr * target_mult)
        
        stop = entry_price - s_dist if action == "BUY" else entry_price + s_dist
        target = entry_price + t_dist if action == "BUY" else entry_price - t_dist
        
        exit_price = None
        
        # Vectorized Intra-day Check
        # For performance, we assume OHLC check is enough.
        # Strict Mode: Iterate minute bars
        for idx, row in day_data.iterrows():
            if action == "BUY":
                if row['Low'] <= stop: exit_price = stop; break
                if row['High'] >= target: exit_price = target; break
            elif action == "SELL":
                if row['High'] >= stop: exit_price = stop; break
                if row['Low'] <= target: exit_price = target; break
        
        if exit_price is None:
            exit_price = day_data.iloc[-1]['Close']
            
        # PnL Calculation
        diff = (exit_price - entry_price) if action == "BUY" else (entry_price - exit_price)
        
        # PnL = (Diff * Multiplier * Lots) - Cost
        bn = (diff * MULTIPLIER * BACKTEST_LOTS) - COST_PER_TRADE
        
        trades.append(bn)
        capital += bn
        trade_count += 1
        if bn > 0: win_count += 1

    total_profit = sum([x for x in trades if x > 0])
    total_loss = abs(sum([x for x in trades if x < 0]))
    pf = (total_profit / total_loss) if total_loss > 0 else 0
    
    total_ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    avg_monthly_ret = total_ret / months if months > 0 else 0
    avg_monthly_pnl = (capital - INITIAL_CAPITAL) / months if months > 0 else 0
    
    return {
        "return": total_ret,
        "monthly_ret": avg_monthly_ret,
        "monthly_pnl": avg_monthly_pnl,
        "pf": pf, 
        "win_rate": (win_count/trade_count*100) if trade_count else 0,
        "trades": trade_count
    }

def run_grid_search_raptor():
    # 1. Load Data
    df_1m = load_and_merge_data()
    # 2. create 15m
    df_15m = create_session_data(df_1m)
    
    # Pre-calc daily ATR
    df_daily = df_1m.resample('D').agg({'High':'max','Low':'min','Close':'last'}).dropna()
    df_daily['ATR'] = (df_daily['High'] - df_daily['Low']).rolling(14).mean()
    
    print(f"🔎 Raptor225 Micro Simulation ({BACKTEST_LOTS} Lots, Cost={COST_PER_TRADE}JPY)")
    print(f"   Period: 2018-2025 | Logic: Stop {STOP_RANGE} / Target {TARGET_RANGE}")
    print("="*100)
    print("Stop | Tgt  || Ret%   | Avg/Mo% | Avg/Mo(JPY) | PF   | Win%  | Trades")
    print("-" * 100)
    
    for s_mult, t_mult in itertools.product(STOP_RANGE, TARGET_RANGE):
        res = run_raptor_simulation_risk(df_1m, df_15m, BEST_G, BEST_N, s_mult, t_mult, df_daily)
        print(f"{s_mult:<4} | {t_mult:<4} || {res['return']:>6.1f}% | {res['monthly_ret']:>6.2f}% | ¥{res['monthly_pnl']:>9,.0f} | {res['pf']:4.2f} | {res['win_rate']:5.1f}% | {res['trades']}")
        
    print("="*100)

if __name__ == "__main__":
    run_grid_search_raptor()
