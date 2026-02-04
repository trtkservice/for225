import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration
INITIAL_CAPITAL = 100000
START_DATE = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
CONTRACT_MULTIPLIER = 10
COST_PER_TRADE = 50
TICK_SIZE = 5

# Risk Params (Optimized)
STOP_ATR_MULT = 0.6
TARGET_ATR_MULT = 1.2

def round_to_tick(price):
    return round(price / TICK_SIZE) * TICK_SIZE

def calc_indicators(df):
    # Trend (EMA)
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # Momentum (RSI - Daily as proxy for intraday)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd - signal
    
    # Volatility (ATR)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    return df

def get_signal(row, vix_val):
    # Antigravity Logic Reproduction
    
    # 1. Trend Score
    trend_val = 0
    p = row['Close']
    if p > row['EMA20']: trend_val += 0.3
    if p > row['EMA50']: trend_val += 0.3
    if p > row['EMA200']: trend_val += 0.4
    if row['EMA20'] > row['EMA50']: trend_val += 0.2
    
    if p < row['EMA20']: trend_val -= 0.3
    if p < row['EMA50']: trend_val -= 0.3
    if p < row['EMA200']: trend_val -= 0.4
    if row['EMA20'] < row['EMA50']: trend_val -= 0.2
    
    trend_score = np.clip(trend_val, -1.0, 1.0)
    
    # 2. Momentum Score (Daily Proxy)
    mom_val = 0
    rsi = row['RSI']
    if rsi > 70: mom_val -= 0.3
    elif rsi < 30: mom_val += 0.3
    elif rsi > 50: mom_val += 0.2
    else: mom_val -= 0.2
    
    if row['MACD_Hist'] > 0: mom_val += 0.4
    else: mom_val -= 0.4
    
    mom_score = np.clip(mom_val, -1.0, 1.0)
    
    # 3. Volatility Score
    vol_val = 0
    if vix_val < 15: vol_val += 0.2
    elif vix_val > 30: vol_val -= 0.8
    elif vix_val > 20: vol_val -= 0.3
    vol_score = vol_val
    
    # Integration
    total = (trend_score * 0.4) + (mom_score * 0.4) + (vol_score * 0.2)
    
    if total > 0.3: return "LONG"
    elif total < -0.3: return "SHORT"
    return "WAIT"

def run_backtest():
    print(f"📥 Fetching 3 years of data (Start: {START_DATE})...")
    
    # Fetch Data
    nikkei = yf.download("^N225", start=START_DATE, progress=False) 
    vix = yf.download("^VIX", start=START_DATE, progress=False)
    
    # Flatten MultiIndex columns if present (Fix for recent yfinance)
    if isinstance(nikkei.columns, pd.MultiIndex):
        nikkei.columns = nikkei.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
        
    # Preprocess
    nikkei = calc_indicators(nikkei)
    
    # Align VIX
    vix = vix['Close'].reindex(nikkei.index).fillna(20.0)
    
    capital = INITIAL_CAPITAL
    trades = []
    equity_curve = [capital]
    
    print("RUNNING SIMULATION...")
    
    for i in range(200, len(nikkei)-1):
        # Extract scalar values explicitly to avoid Series comparison error
        today_row = nikkei.iloc[i]
        today_close = float(today_row['Close'])
        today_ema20 = float(today_row['EMA20'])
        today_ema50 = float(today_row['EMA50'])
        today_ema200 = float(today_row['EMA200'])
        
        # Pass dictionary with scalars instead of Series
        idx_data = {
            'Close': today_close,
            'EMA20': today_ema20,
            'EMA50': today_ema50,
            'EMA200': today_ema200,
            'RSI': float(today_row['RSI']),
            'MACD_Hist': float(today_row['MACD_Hist']),
            'ATR': float(today_row['ATR']) if not pd.isna(today_row['ATR']) else 400.0
        }
        
        vix_val = float(vix.iloc[i])
        
        # 1. Signal
        signal = get_signal(idx_data, vix_val)
        
        if signal == "WAIT":
            equity_curve.append(capital)
            continue

        # 2. Setup Trade
        tomorrow = nikkei.iloc[i+1] # Trading Day remains DataFrame row
        atr = idx_data['ATR'] # Usescalar from dict
        
        entry_price = round_to_tick(float(tomorrow['Open'])) # Enter at Open
        stop_dist = round_to_tick(atr * STOP_ATR_MULT)
        target_dist = round_to_tick(atr * TARGET_ATR_MULT)
        
        stop_price = entry_price - stop_dist if signal == "LONG" else entry_price + stop_dist
        target_price = entry_price + target_dist if signal == "LONG" else entry_price - target_dist
        
        # 3. Sim Execution (Intraday approx check)
        t_high = tomorrow['High']
        t_low = tomorrow['Low']
        t_close = tomorrow['Close']
        
        hit_stop = False
        hit_target = False
        
        if signal == "LONG":
            if t_low <= stop_price: hit_stop = True
            if t_high >= target_price: hit_target = True
        else:
            if t_high >= stop_price: hit_stop = True
            if t_low <= target_price: hit_target = True
            
        # Outcome Logic
        exit_price = t_close
        if hit_stop:
            exit_price = stop_price
        elif hit_target:
            exit_price = target_price
            
        # PnL
        p_diff = (exit_price - entry_price) if signal == "LONG" else (entry_price - exit_price)
        gross = p_diff * CONTRACT_MULTIPLIER
        net = gross - COST_PER_TRADE
        
        capital += net
        trades.append(net)
        equity_curve.append(capital)
        
    # Results
    total_trades = len(trades)
    wins = len([t for t in trades if t > 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    print("\n" + "="*40)
    print(f"🏁 BACKTEST RESULT (3 Years)")
    print("="*40)
    print(f"💰 Final Capital:  ¥{capital:,.0f}")
    print(f"📈 Total Return:   {total_return:+.2f}%")
    print(f"📊 Win Rate:       {win_rate:.1f}% ({wins}/{total_trades})")
    print(f"💸 Total Trades:   {total_trades}")
    print("="*40)
    
    # Save chart data if needed (skipped for now)

if __name__ == "__main__":
    run_backtest()
