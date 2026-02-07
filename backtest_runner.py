
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
try:
    from nikkei_bot import Config, round_to_tick
except ImportError:
    class Config:
        TICK_SIZE = 5
    def round_to_tick(price):
        return int(round(price / 5) * 5)

# Data Directory
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Simulation Config (Strict Settings for Micro 2 Lots)
INITIAL_CAPITAL = 100000 # 100k start
BACKTEST_LOTS = 2        # 2 Micro Lots
MULTIPLIER = 10          # Micro = 10x
COST_PER_TRADE = 60      # Rakuten Micro ~11JPY x 2 x 2 sides + Slippage

# --- Raptor EXACT Settings (from Internal Prompt) ---
BEST_G = 0.0025   # 0.25%
BEST_N = 32       # 32 bars (8 hours) - Changed from 48
BEST_R = 1.8      # Overheat threshold

# Search Range for Risk Management
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
    Create Session Data (15m Data).
    """
    print("🔄 Creating Session & 15m Data...")
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

def precalculate_night_ranges(df_1m):
    """
    Calculate the High-Low range for every Night Session.
    Night Session T-1 (16:30) to T (06:00) is attributed to Date T.
    """
    print("🌙 Pre-calculating Night Session Ranges (Logic D)...")
    night_ranges = {}
    
    unique_dates = sorted(list(set(df_1m.index.date)))
    
    for i in range(1, len(unique_dates)):
        curr_date = unique_dates[i]
        prev_date = unique_dates[i-1]
        
        night_start = datetime.combine(prev_date, time(16, 30))
        night_end = datetime.combine(curr_date, time(6, 0))
        
        # Slicing
        night_data = df_1m.loc[night_start:night_end]
        if night_data.empty:
            continue
            
        rng = night_data['High'].max() - night_data['Low'].min()
        night_ranges[curr_date] = rng
        
    return night_ranges

def run_raptor_simulation_risk(df_1m, df_15m, g_cut, n_period, r_factor, stop_mult, target_mult, df_daily, night_ranges):
    capital = INITIAL_CAPITAL
    trades = []
    
    # Debug Metrics
    atr_values = []
    d_activations = 0
    
    unique_dates = sorted(list(set(df_1m.index.date))) # Dates
    # Convert night_ranges keys to list for indexing if needed, or use dict
    # We need "Past 10 sessions average". 
    # Let's create a DataFrame for sliding window
    nr_df = pd.DataFrame.from_dict(night_ranges, orient='index', columns=['Range'])
    nr_df['Avg10'] = nr_df['Range'].rolling(10).mean().shift(1) # Avg of LAST 10 (excluding current)
    # Actually prompt says: "Avg(Last 10) including current? or excluding?"
    # Prompt: "Review prev session range vs Avg(Last 10)". Usually implies historical baseline.
    # Let's use rolling mean of previous 10.
    
    months = len(set([d.strftime('%Y-%m') for d in unique_dates]))
    trades_executed = 0
    wins = 0
    
    print(f"🚀 Starting Exact Raptor Simulation... (Dates: {len(unique_dates)})")
    
    for i in range(10, len(unique_dates)): # Start from 10 to allow history
        curr_date = unique_dates[i]
        prev_date = unique_dates[i-1]
        
        # --- RAPTOR LOGIC START ---
        
        day_start = datetime.combine(curr_date, time(8, 45))
        day_end = datetime.combine(curr_date, time(15, 15))
        
        day_data = df_1m.loc[day_start:day_end]
        if day_data.empty: continue
        entry_price = round_to_tick(day_data.iloc[0]['Open'])
        
        night_start = datetime.combine(prev_date, time(16, 30))
        night_end = datetime.combine(curr_date, time(6, 0))
        night_data = df_1m.loc[night_start:night_end]
        
        if night_data.empty: continue
            
        prev_close = night_data.iloc[-1]['Close']
        prev_open = night_data.iloc[0]['Open']
        
        # 1. Gap Check (RiskGate)
        gap_rate = (entry_price - prev_close) / prev_close
        if abs(gap_rate) >= g_cut: continue
        
        # 2. B: Night Trend
        # 陽線:+1、陰線:-1、同値:0
        if prev_close > prev_open: score_b = 1
        elif prev_close < prev_open: score_b = -1
        else: score_b = 0
        
        # 4. D: Overheat logic
        # Check night range
        is_overheat = False
        if curr_date in nr_df.index:
            current_range = nr_df.loc[curr_date]['Range']
            avg_range = nr_df.loc[curr_date]['Avg10']
            
            if not pd.isna(avg_range) and avg_range > 0:
                if current_range >= (avg_range * r_factor):
                    is_overheat = True
                    d_activations += 1
                    # Weaken B
                    score_b = 0
        
        # 3. C: Momentum
        # Last N=32 bars ENDING at day_start
        recent_15m = df_15m.loc[night_start:day_start].iloc[:-1] # Exclude 08:45 candle
        recent_15m = recent_15m.iloc[-n_period:]
        if len(recent_15m) < n_period * 0.5: continue
            
        slope = calculate_slope(recent_15m['Close'])
        score_c = 1 if slope > 0 else -1
        
        # Total
        total = score_b + score_c
        action = "NO-TRADE"
        if total >= 2: action = "BUY"
        elif total <= -2: action = "SELL"
        
        if action == "NO-TRADE": continue
        
        # --- RISK EXECUTION ---
        atr = 800.0 
        try:
            ts_lookup = pd.Timestamp(prev_date)
            idx = df_daily.index.get_indexer([ts_lookup], method='pad')[0]
            if idx != -1:
                atr_cand = df_daily.iloc[idx]['ATR']
                if not pd.isna(atr_cand) and atr_cand > 0: atr = atr_cand
        except: pass
        atr_values.append(atr)
        
        s_dist = int(round(atr * stop_mult / Config.TICK_SIZE)) * Config.TICK_SIZE
        t_dist = int(round(atr * target_mult / Config.TICK_SIZE)) * Config.TICK_SIZE
        
        stop = round_to_tick(entry_price - s_dist) if action == "BUY" else round_to_tick(entry_price + s_dist)
        target = round_to_tick(entry_price + t_dist) if action == "BUY" else round_to_tick(entry_price - t_dist)
        
        exit_price = None
        for idx, row in day_data.iterrows():
            if action == "BUY":
                if row['Low'] <= stop: exit_price = stop; break
                if row['High'] >= target: exit_price = target; break
            elif action == "SELL":
                if row['High'] >= stop: exit_price = stop; break
                if row['Low'] <= target: exit_price = target; break
        
        if exit_price is None:
            exit_price = round_to_tick(day_data.iloc[-1]['Close'])
            
        diff = (exit_price - entry_price) if action == "BUY" else (entry_price - exit_price)
        bn = (diff * MULTIPLIER * BACKTEST_LOTS) - COST_PER_TRADE
        
        trades.append(bn)
        capital += bn
        trades_executed += 1
        if bn > 0: wins += 1

        if trades_executed <= 3:
             print(f"DEBUG #{trades_executed}: {curr_date} {action} B={score_b}(OH={is_overheat}) C={score_c} PnL={bn:.0f}")

    total_profit = sum([x for x in trades if x > 0])
    total_loss = abs(sum([x for x in trades if x < 0]))
    pf = (total_profit / total_loss) if total_loss > 0 else 0
    total_ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    avg_monthly_pnl = (capital - INITIAL_CAPITAL) / months if months > 0 else 0
    
    print("-" * 50)
    print(f"DEBUG: D-Logic triggered {d_activations} times.")
    print("-" * 50)
    
    return {
        "return": total_ret,
        "monthly_pnl": avg_monthly_pnl,
        "pf": pf, 
        "win_rate": (wins/trades_executed*100) if trades_executed else 0,
        "trades": trades_executed
    }

def run_grid_search_raptor():
    df_1m = load_and_merge_data()
    df_15m = create_session_data(df_1m)
    
    # Pre-calc daily ATR
    df_daily = df_1m.resample('D').agg({'High':'max','Low':'min','Close':'last'}).dropna()
    df_daily['ATR'] = (df_daily['High'] - df_daily['Low']).rolling(14).mean()
    
    # Pre-calc Night Ranges for D-Logic
    night_ranges = precalculate_night_ranges(df_1m)
    
    print(f"🔎 Raptor225 EXACT (Micro 2 Lots, N={BEST_N}, D-Logic)")
    print("="*100)
    print("Stop | Tgt  || Ret%   | Avg/Mo(JPY) | PF   | Win%  | Trades")
    print("-" * 100)
    
    for s_mult, t_mult in itertools.product(STOP_RANGE, TARGET_RANGE):
        res = run_raptor_simulation_risk(df_1m, df_15m, BEST_G, BEST_N, BEST_R, s_mult, t_mult, df_daily, night_ranges)
        print(f"{s_mult:<4} | {t_mult:<4} || {res['return']:>6.1f}% | ¥{res['monthly_pnl']:>9,.0f} | {res['pf']:4.2f} | {res['win_rate']:5.1f}% | {res['trades']}")
    print("="*100)

if __name__ == "__main__":
    run_grid_search_raptor()
