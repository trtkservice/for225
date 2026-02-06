
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
from nikkei_bot import Config, TechnicalAnalysis, LiLFlexxEngine, round_to_tick

# Data Directory
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Simulation Config
INITIAL_CAPITAL = 100000
BACKTEST_LOTS = 1
SPREAD = 5.0
COST_PER_TRADE = 75

# --- Parameters for Grid Search ---
STOP_RANGE = [0.4, 0.5, 0.6, 0.7, 0.8]
TARGET_RANGE = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

def load_and_merge_data():
    """Load all N225minif_*.xlsx files and merge them."""
    pattern = os.path.join(DATA_DIR, "N225minif_*.xlsx")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("❌ No excel files found matching 'N225minif_*.xlsx'")
        sys.exit(1)
        
    print(f"📥 Loading {len(files)} Excel files... (This may take a minute)")
    
    df_list = []
    first_file = True
    for f in files:
        print(f"   Reading {os.path.basename(f)}...")
        try:
            temp = pd.read_excel(f)
            if first_file:
                print(f"👀 Columns in {os.path.basename(f)}: {temp.columns.tolist()}")
                # sys.exit(0) # Don't exit yet, let's try to map dynamically if possible, but exiting is safer to see logs.
                # Actually, let's just print and let it fail, so we can see the logs.
                first_file = False
            df_list.append(temp)
        except Exception as e:
            print(f"   ⚠️ Failed to read {f}: {e}")

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"👀 Full DF Columns: {full_df.columns.tolist()}")
    
    # Rename map (adjust based on actual file headers if needed)
    rename_map = {
        '日付': 'Date', 'Date': 'Date', 'date': 'Date',
        '時刻': 'Time', 'Time': 'Time', 'time': 'Time',
        '始値': 'Open', 'Open': 'Open', 'open': 'Open',
        '高値': 'High', 'High': 'High', 'high': 'High',
        '安値': 'Low', 'Low': 'Low', 'low': 'Low',
        '終値': 'Close', 'Close': 'Close', 'close': 'Close'
    }
    full_df.rename(columns=rename_map, inplace=True)
    print(f"👀 Renamed Columns: {full_df.columns.tolist()}")
    
    # Combine Date+Time to Datetime Index
    # Note: 'Date' might be string or datetime. 'Time' might be string/time object.
    
    def parse_datetime(row):
        d = row['Date']
        t = row['Time']
        if isinstance(d, str): d = datetime.strptime(d, '%Y/%m/%d').date() # Guess format
        if isinstance(d, datetime): d = d.date()
        
        if isinstance(t, str):
            t = datetime.strptime(t, '%H:%M').time() # Guess format
        
        return datetime.combine(d, t)

    print("   Processing timestamps...")
    # Fast vectorized conversion if possible, but safe fallback
    try:
        # Optimistic: Date is datetime64, Time is time object
        # If excel parsed date correctly, Date col is datetime.
        # Check first row
        full_df['Datetime'] = pd.to_datetime(full_df['Date'].astype(str) + ' ' + full_df['Time'].astype(str))
    except:
        # Fallback to slower row-wise if formats are complex
        full_df['Datetime'] = full_df.apply(parse_datetime, axis=1)

    full_df.set_index('Datetime', inplace=True)
    full_df.sort_index(inplace=True)
    
    # Drop duplicates
    full_df = full_df[~full_df.index.duplicated(keep='first')]
    
    # Keep only OHLC
    df = full_df[['Open', 'High', 'Low', 'Close']].astype(float)
    
    print(f"✅ Loaded {len(df)} 1-minute bars ({df.index[0]} to {df.index[-1]})")
    return df

def resample_data(df_1m):
    """Create Daily and 15m datasets."""
    # 15m bars
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    
    # Daily bars (Day Session Only logic is complex, for now assume standard 24h split at 00:00 or 15:15?)
    # For BacktestEngine (signal gen), we usually use 24h Daily candles or Day Session?
    # Let's use standard Daily (00:00-24:00) for simplicity of signal generation compatibility with yfinance
    df_daily = df_1m.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    
    # Calc ATR on Daily
    df_daily['ATR'] = TechnicalAnalysis.calc_atr(df_daily)
    
    return df_daily, df_15m

class FastBacktestEngine(LiLFlexxEngine):
    """Engine optimized for pre-calced data."""
    def __init__(self, daily_df):
        # We only pass daily for signal gen
        self.data = {"nikkei_futures_daily": daily_df}
        self.scores = {}
        
    def _analyze_momentum(self):
        # Simplified momentum for daily proxy
        df = self.data["nikkei_futures_daily"]
        if df is None or len(df) < 50: 
            self.scores["momentum"] = 0
            self.scores["details"] = {"rsi": 50}
            return

        close = df['Close']
        rsi = TechnicalAnalysis.calc_rsi(close).iloc[-1]
        self.scores["details"] = {"rsi": round(rsi, 1) if not np.isnan(rsi) else 50}
        
        val = 0
        if rsi > 70: val -= 0.3
        elif rsi < 30: val += 0.3
        elif rsi > 50: val += 0.2
        else: val -= 0.2
        
        self.scores["momentum"] = round(np.clip(val, -1.0, 1.0), 3)

def generate_signals(df_daily):
    """Pre-calculate signals for ALL days."""
    print("🚦 Generating Signals...")
    signals = {} # Date -> Signal, ATR
    
    # Need some buffer for indicators
    for i in range(50, len(df_daily)-1):
        window = df_daily.iloc[i-50:i+1].copy()
        
        engine = FastBacktestEngine(window)
        scores = engine.analyze()
        sig = scores['signal']
        strength = scores['strength']
        
        # Filter WEAK signals if you want strict logic
        # if strength == "WEAK": sig = "WAIT"
        
        # Store for Next Day's trade
        next_date = df_daily.index[i+1].date()
        atr = df_daily.iloc[i]['ATR']
        if pd.isna(atr): atr = 300.0
        
        signals[next_date] = {'type': sig, 'atr': atr}
        
    return signals

def run_intraday_simulation(df_1m, signals, stop_mult, target_mult):
    """
    Run simulation using 1-minute data for precise Stop/Target/Close execution.
    Mode is fixed to 'DAY' (close at 15:15).
    """
    capital = INITIAL_CAPITAL
    trades = []
    
    # Loop continuously? No, iterate by Day
    # Filter 1m data to only days we have signals for
    # Group by Date
    
    # Optimization: Iterate over signal dates
    grouped = df_1m.groupby(df_1m.index.date)
    
    for date, group in grouped:
        if date not in signals: continue
        sig_data = signals[date]
        signal = sig_data['type']
        
        if signal == "WAIT": continue
        
        # Day Session Only (08:45 - 15:15)
        # Filter group for these hours
        # group is already datetime indexed
        
        # 1. Determine Entry at Day Open (08:45 or first bar)
        # We assume we enter at the Open of the first bar of the day session
        # Check time. 08:45 ~ 15:15
        day_session = group.between_time('08:45', '15:15')
        
        if day_session.empty: continue
        
        first_bar = day_session.iloc[0]
        entry_price = round_to_tick(first_bar['Open'])
        
        # Calc Stop/Target
        atr = sig_data['atr']
        s_dist = round_to_tick(atr * stop_mult)
        t_dist = round_to_tick(atr * target_mult)
        
        stop = entry_price - s_dist if signal == "LONG" else entry_price + s_dist
        target = entry_price + t_dist if signal == "LONG" else entry_price - t_dist
        
        # 2. Iterate 1m bars to check hit
        # Vectorized check is possible but simple iteration is reliable
        
        exit_price = None
        exit_reason = None
        
        for idx, row in day_session.iterrows():
            high_p = row['High']
            low_p = row['Low']
            close_p = row['Close']
            
            if signal == "LONG":
                # Check Low vs Stop first? Or High vs Target?
                # On a 1m bar, it's ambiguous. But much less than Daily.
                # Assume conservative: Check Low (Stop) first.
                if low_p <= stop:
                    exit_price = stop
                    exit_reason = "STOP"
                    break
                if high_p >= target:
                    exit_price = target
                    exit_reason = "TARGET"
                    break
                    
            elif signal == "SHORT":
                # Check High (Stop) first
                if high_p >= stop:
                    exit_price = stop
                    exit_reason = "STOP"
                    break
                # Spread check: Ask hits stop
                if high_p + SPREAD >= stop:
                    exit_price = stop
                    exit_reason = "STOP (Spread)"
                    break
                
                if low_p <= target:
                    exit_price = target
                    exit_reason = "TARGET"
                    break
        
        # 3. If no hit, exit at Close (15:15)
        if exit_price is None:
            last_bar = day_session.iloc[-1]
            exit_price = last_bar['Close']
            exit_reason = "CLOSE"
            
        # Calc PnL
        diff = (exit_price - entry_price) if signal == "LONG" else (entry_price - exit_price)
        bn = (diff * Config.CONTRACT_MULTIPLIER * BACKTEST_LOTS) - (COST_PER_TRADE * BACKTEST_LOTS)
        
        capital += bn
        trades.append(bn)
        
    # Stats
    total_profit = sum([x for x in trades if x > 0])
    total_loss = abs(sum([x for x in trades if x < 0]))
    pf = (total_profit / total_loss) if total_loss > 0 else 0
    
    return {
        "return": (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "pf": pf,
        "max_dd": 0, # Calculation omitted for speed in grid
        "trades": len(trades)
    }

def run_grid_search_intraday():
    # 1. Load Data
    df_1m = load_and_merge_data()
    df_daily, _ = resample_data(df_1m)
    
    print(f"✅ Data Ready: {len(df_daily)} days")
    
    # 2. Generate Signals
    signals = generate_signals(df_daily)
    
    # 3. Grid Search
    print(f"🔎 Intraday Grid Search (Strict 1-minute execution)")
    print(f"   Period: {df_daily.index[0].date()} - {df_daily.index[-1].date()}")
    print("="*60)
    print("S | T   || Ret%   | PF   | Trades")
    print("-" * 60)
    
    best_ret = -999
    best_set = None
    
    for s, t in itertools.product(STOP_RANGE, TARGET_RANGE):
        if t <= s: continue
        
        res = run_intraday_simulation(df_1m, signals, s, t)
        
        print(f"{s:<3} {t:<3} || {res['return']:>6.1f}% | {res['pf']:4.2f} | {res['trades']}")
        
        if res['return'] > best_ret:
            best_ret = res['return']
            best_set = (s, t)
            
    print("="*60)
    print(f"🏆 Best Setting: Stop {best_set[0]} / Target {best_set[1]} (Ret: {best_ret:.1f}%)")

if __name__ == "__main__":
    run_grid_search_intraday()
