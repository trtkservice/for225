
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

def run_raptor_simulation(df_1m, df_15m, g_cut, n_period, r_factor):
    """
    Run Raptor225 Logic Simulation.
    Target: Day Session (08:45-15:15)
    """
    capital = INITIAL_CAPITAL
    trades = []
    
    # Group by Date
    unique_dates = sorted(list(set(df_1m.index.date)))
    
    trade_count = 0
    win_count = 0
    
    for i in range(1, len(unique_dates)):
        curr_date = unique_dates[i]
        prev_date = unique_dates[i-1]
        
        # 1. Get Day Session Data (Today)
        day_start = datetime.combine(curr_date, time(8, 45))
        day_end = datetime.combine(curr_date, time(15, 15))
        day_data = df_1m.loc[day_start:day_end]
        
        if day_data.empty: continue
        
        # Entry Price (08:45 Open)
        entry_price = day_data.iloc[0]['Open']
        
        # 2. Get Prev Session Data (NIGHT: Yesterday 16:30 - Today 06:00)
        night_start = datetime.combine(prev_date, time(16, 30))
        night_end = datetime.combine(curr_date, time(6, 0)) # approx
        
        night_data = df_1m.loc[night_start:night_end]
        
        if night_data.empty:
            # If no night session (Monday?), fallback to Prev DAY Close?
            # Or skip. Raptor says "Prev Session Close".
            # For simplicity, if no NIGHT data, skip (insufficient data).
            continue
            
        prev_close = night_data.iloc[-1]['Close']
        prev_open = night_data.iloc[0]['Open']
        
        # --- LOGIC START ---
        
        # 4) RiskGate (Gap)
        # gap_rate = (expected_open_price - prev_session_close) / prev_session_close
        gap_rate = (entry_price - prev_close) / prev_close
        if abs(gap_rate) >= g_cut:
            # NO-TRADE (RiskGate Fail)
            continue
            
        # 5) Direction Indicators
        
        # B: Prev Session Direction
        # +1 if Yang (Close > Open), -1 if Yin (Close < Open)
        score_b = 0
        if prev_close > prev_open: score_b = 1
        elif prev_close < prev_open: score_b = -1
        
        # D: Overheat Check
        # Current Night Range vs Avg Night Range (N=10 sessions)
        # We need historical night ranges. For speed, calculate on the fly or pre-calc?
        # On-the-fly approximation: just check Range vs ATR? 
        # Or skip D for now to strictly follow "Simple Raptor". 
        # Let's Skip D first to see baseline. (Assuming r_factor is high enough)
        # Or simpler: if range > 1.8 * prev_range
        night_range = night_data['High'].max() - night_data['Low'].min()
        # To implement D correctly requires valid history. Let's omit D for MVP.
        
        # C: Momentum (15m N periods)
        # Get 15m data ending before 08:45 Today
        cutoff_time = day_start
        recent_15m = df_15m.loc[:cutoff_time].iloc[-(n_period+1):-1] # Exclude exactly 08:45 bar
        
        if len(recent_15m) < n_period: continue
        
        slope = calculate_slope(recent_15m['Close'])
        score_c = 0
        if slope > 0.05: score_c = 1 # Threshold for slope? Or just > 0?
        elif slope < -0.05: score_c = -1
        # Raptor prompt: "Positive:+1, Negative:-1, Tiny:0"
        # Let's assume strict signs for now.
        if slope > 0: score_c = 1
        elif slope < 0: score_c = -1
        
        # Total Score
        total_score = score_b + score_c
        
        # Verdict
        action = "NO-TRADE"
        if total_score >= 2: action = "BUY"
        elif total_score <= -2: action = "SELL"
        
        if action == "NO-TRADE": continue
        
        # --- EXECUTION (Simulated) ---
        
        # Exit: Close (15:15)
        exit_price = day_data.iloc[-1]['Close']
        
        # PnL
        diff = (exit_price - entry_price) if action == "BUY" else (entry_price - exit_price)
        bn = (diff * 100) - 0 # Lots=100 (mini), Cost=0 as per Raptor spec
        
        trades.append(bn)
        capital += bn
        trade_count += 1
        if bn > 0: win_count += 1
        
    # Stats
    total_profit = sum([x for x in trades if x > 0])
    total_loss = abs(sum([x for x in trades if x < 0]))
    pf = (total_profit / total_loss) if total_loss > 0 else 0
    win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0
    ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    return {
        "return": ret,
        "pf": pf,
        "trades": trade_count,
        "win_rate": win_rate
    }

def run_grid_search_raptor():
    # 1. Load Data
    df_1m = load_and_merge_data()
    # 2. create 15m
    df_15m = create_session_data(df_1m)
    
    print(f"🔎 Raptor225 Grid Search")
    print(f"   Period: 2018-2025 | Logic: Gap < Cut, B+C >= 2 -> Trade")
    print("="*80)
    print("G_cut   | N  || Ret%   | PF   | Win%  | Trades")
    print("-" * 80)
    
    for g_cut, n_period in itertools.product(G_CUT_RANGE, N_RANGE):
        res = run_raptor_simulation(df_1m, df_15m, g_cut, n_period, R_FACTOR)
        print(f"{g_cut:.4f} | {n_period:<2} || {res['return']:>6.1f}% | {res['pf']:4.2f} | {res['win_rate']:5.1f}% | {res['trades']}")
        
    print("="*80)

if __name__ == "__main__":
    run_grid_search_raptor()
