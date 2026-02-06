
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
INITIAL_CAPITAL = 100000
BACKTEST_LOTS = 1
SPREAD = 0 # Raptor assumes mid price or handled in cost? Let's use 0 for raw logic check, or 5 if strict.
COST_PER_TRADE = 0 # Raptor prompt says cost=0 fixed. We can add later.

# --- Raptor Grid Parameters ---
# G_cut: Gap threshold (default 0.0025)
# N: Momentum period (default 32)
G_CUT_RANGE = [0.0025, 0.0050, 0.0075] 
N_RANGE = [16, 32, 48]
# Overheat threshold r
R_FACTOR = 1.8

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
    
    # Rename map (Corrected)
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
    DAY: 08:45 - 15:15
    NIGHT: 16:30 - Next 06:00 (approx)
    """
    print("🔄 Creating Session & 15m Data...")
    
    # 1. 15m Data
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    
    # 2. Session Data
    # Assign 'Session' label to each row
    # DAY: 08:45 <= t <= 15:15
    # NIGHT: 16:30 <= t <= 23:59 OR 00:00 <= t <= 06:00
    
    # Initialize 'Session' col
    # We need to group NIGHT sessions that span across midnight to the 'Trade Date'
    # Actually, Raptor logic uses "Prev Session".
    # For DAY trade (Trade Date T), Prev Session is NIGHT of T (ends 06:00 T).
    # NIGHT of T starts 16:30 T-1.
    
    # Simplification: We iterate each Day (T).
    # We identify NIGHT preceeding T (from 16:30 T-1 to 06:00 T)
    # We identify DAY of T (08:45 T to 15:15 T)
    
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

# --- Raptor Best Settings ---
BEST_G = 0.0025
BEST_N = 48
BEST_R = 1.8

# Search Range for Risk Management
STOP_RANGE = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
TARGET_RANGE = [1.0, 2.0, 3.0, 4.0, 5.0, 99.0] # 99.0 = Almost No Target

def run_raptor_with_risk_grid(df_1m, df_15m, g_cut=BEST_G, n_period=BEST_N):
    # Pre-calculate Sessions (Optimization)
    unique_dates = sorted(list(set(df_1m.index.date)))
    
    # Pre-calc daily ATR for dynmic stop/target
    df_daily = df_1m.resample('D').agg({'High':'max','Low':'min','Close':'last'}).dropna()
    df_daily['ATR'] = (df_daily['High'] - df_daily['Low']).rolling(14).mean() # Simplified ATR
    
    signals = [] # List of (Date, Action, ATR)
    
    # 1. Generate Signals First (Avoid re-calc in loop)
    for i in range(1, len(unique_dates)):
        curr_date = unique_dates[i]
        prev_date = unique_dates[i-1]
        
        # Data slices (Same logic as before)
        day_start = datetime.combine(curr_date, time(8, 45))
        prev_night_start = datetime.combine(prev_date, time(16, 30))
        cutoff_time = day_start
        
        # Quick checks to skip if data missing
        if day_start not in df_1m.index: continue # Approx check
        
        # Need actual data access, slow loop but necessary for accuracy
        # ... (Signal logic omitted here for brevity, assume we use the same logic block)
        # To save code space, we merge logic into the main simulation loop below
        pass 

    print(f"🔎 Raptor225 Risk Management Grid (G={g_cut}, N={n_period})")
    print("="*80)
    print("Stop | Tgt  || Ret%   | PF   | Win%  | Trades")
    print("-" * 80)

    for s_mult, t_mult in itertools.product(STOP_RANGE, TARGET_RANGE):
        res = run_raptor_simulation_risk(df_1m, df_15m, g_cut, n_period, s_mult, t_mult, df_daily)
        print(f"{s_mult:<4} | {t_mult:<4} || {res['return']:>6.1f}% | {res['pf']:4.2f} | {res['win_rate']:5.1f}% | {res['trades']}")
    print("="*80)

def run_raptor_simulation_risk(df_1m, df_15m, g_cut, n_period, stop_mult, target_mult, df_daily):
    capital = INITIAL_CAPITAL
    trades = []
    win_count = 0
    trade_count = 0
    
    unique_dates = sorted(list(set(df_1m.index.date)))
    
    for i in range(1, len(unique_dates)):
        curr_date = unique_dates[i]
        prev_date = unique_dates[i-1]
        
        # --- RAPTOR LOGIC (Repeated for each grid, could be optimized but fast enough) ---
        day_start = datetime.combine(curr_date, time(8, 45))
        day_end = datetime.combine(curr_date, time(15, 15))
        
        day_data = df_1m.loc[day_start:day_end]
        if day_data.empty: continue
        entry_price = day_data.iloc[0]['Open']
        
        night_start = datetime.combine(prev_date, time(16, 30))
        night_end = datetime.combine(curr_date, time(6, 0))
        night_data = df_1m.loc[night_start:night_end]
        
        if night_data.empty: continue
        prev_close = night_data.iloc[-1]['Close']
        prev_open = night_data.iloc[0]['Open']
        
        # Gap
        gap_rate = (entry_price - prev_close) / prev_close
        if abs(gap_rate) >= g_cut: continue
        
        # B
        score_b = 1 if prev_close > prev_open else -1
        
        # C
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
        # Get ATR
        atr = 300 # Default
        try:
            atr = df_daily.loc[str(prev_date)]['ATR']
            if pd.isna(atr): atr = 300
        except: pass
        
        s_dist = round_to_tick(atr * stop_mult)
        t_dist = round_to_tick(atr * target_mult)
        
        stop = entry_price - s_dist if action == "BUY" else entry_price + s_dist
        target = entry_price + t_dist if action == "BUY" else entry_price - t_dist
        
        # Intra-bar check
        exit_price = None
        
        # Vectorized check for day
        # For BUY: Low < Stop? High > Target?
        # For SELL: High > Stop? Low < Target?
        
        # We need precise timing.
        for idx, row in day_data.iterrows():
            if action == "BUY":
                if row['Low'] <= stop: exit_price = stop; break
                if row['High'] >= target: exit_price = target; break
            elif action == "SELL":
                if row['High'] >= stop: exit_price = stop; break
                if row['Low'] <= target: exit_price = target; break
        
        if exit_price is None:
            exit_price = day_data.iloc[-1]['Close']
            
        diff = (exit_price - entry_price) if action == "BUY" else (entry_price - exit_price)
        bn = (diff * 100)
        
        trades.append(bn)
        capital += bn
        trade_count += 1
        if bn > 0: win_count += 1

    total_profit = sum([x for x in trades if x > 0])
    total_loss = abs(sum([x for x in trades if x < 0]))
    pf = (total_profit / total_loss) if total_loss > 0 else 0
    return {
        "return": (capital - INITIAL_CAPITAL)/INITIAL_CAPITAL*100,
        "pf": pf, "win_rate": (win_count/trade_count*100) if trade_count else 0,
        "trades": trade_count
    }

if __name__ == "__main__":
    # Load once
    df_1m = load_and_merge_data()
    df_15m = create_session_data(df_1m)
    run_raptor_with_risk_grid(df_1m, df_15m)
