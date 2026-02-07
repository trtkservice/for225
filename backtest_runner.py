#!/usr/bin/env python3
"""
Raptor225 バックテスト (グリッドサーチ版)
=========================================
楽天証券 / 資金10万円 / 日経225マイクロ1枚 / デイトレード

Stop/Target ATR倍率の最適値を探索する
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from datetime import datetime, time, timedelta
import warnings
import itertools

warnings.filterwarnings('ignore')

# ============================================================
# 運用設定
# ============================================================
CAPITAL = 100_000      # 資金10万円
LOTS = 1               # マイクロ1枚
MULTIPLIER = 10        # 1ポイント = 10円
COMMISSION = 22        # 往復手数料 (11円×2)
TICK = 5               # 呼値

# ============================================================
# Raptorロジック設定
# ============================================================
GAP_CUT = 0.0025       # ギャップ閾値 0.25%

# グリッドサーチ範囲
STOP_RANGE = [0.3, 0.5, 0.7, 1.0]
TARGET_RANGE = [0.5, 1.0, 1.5, 2.0]

# ============================================================
# ユーティリティ
# ============================================================
def tick_round(price):
    return int(round(price / TICK) * TICK)

def calc_slope(closes):
    n = len(closes)
    if n < 2:
        return 0
    x = np.arange(n)
    y = closes.values if hasattr(closes, 'values') else closes
    return np.polyfit(x, y, 1)[0]

# ============================================================
# データ読み込み
# ============================================================
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(base, "N225minif_*.xlsx")))
    
    if not files:
        print("❌ N225minif_*.xlsx が見つかりません")
        sys.exit(1)
    
    print(f"📥 {len(files)}ファイル読み込み中...")
    
    dfs = []
    for f in files:
        print(f"   {os.path.basename(f)}")
        dfs.append(pd.read_excel(f))
    
    df = pd.concat(dfs, ignore_index=True)
    df.rename(columns={
        '日付': 'Date', '時間': 'Time', '時刻': 'Time',
        '始値': 'Open', '高値': 'High', '安値': 'Low', '終値': 'Close'
    }, inplace=True)
    
    df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df = df.set_index('Datetime').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df = df[['Open', 'High', 'Low', 'Close']].astype(float)
    
    print(f"✅ {len(df):,}本 ({df.index[0]} 〜 {df.index[-1]})")
    return df

# ============================================================
# セッションデータ
# ============================================================
def get_session(df_1m, start_dt, end_dt):
    return df_1m.loc[start_dt:end_dt]

def get_session_ohlc(df_1m, start_dt, end_dt):
    s = get_session(df_1m, start_dt, end_dt)
    if s.empty:
        return None
    return {
        'open': s.iloc[0]['Open'],
        'high': s['High'].max(),
        'low': s['Low'].min(),
        'close': s.iloc[-1]['Close']
    }

# ============================================================
# Raptorシグナル
# ============================================================
def raptor_signal(prev_ohlc, slope):
    if prev_ohlc['close'] > prev_ohlc['open']:
        score_b = 1
    elif prev_ohlc['close'] < prev_ohlc['open']:
        score_b = -1
    else:
        score_b = 0
    
    score_c = 1 if slope > 0 else -1
    total = score_b + score_c
    
    if total >= 2:
        return 'BUY'
    elif total <= -2:
        return 'SELL'
    return None

# ============================================================
# トレード実行
# ============================================================
def execute_trade(df_1m, action, entry_price, stop, target, session_end):
    session_data = df_1m.loc[:session_end]
    
    if session_data.empty:
        return entry_price, 'NO_DATA'
    
    for _, bar in session_data.iterrows():
        if action == 'BUY':
            if bar['Low'] <= stop:
                return stop, 'STOP'
            if bar['High'] >= target:
                return target, 'TARGET'
        else:
            if bar['High'] >= stop:
                return stop, 'STOP'
            if bar['Low'] <= target:
                return target, 'TARGET'
    
    return tick_round(session_data.iloc[-1]['Close']), 'CLOSE'

# ============================================================
# バックテスト (1パラメータセット)
# ============================================================
def run_backtest(df_1m, df_15m, df_daily, dates, stop_mult, target_mult):
    capital = CAPITAL
    trades = []
    
    for i in range(1, len(dates)):
        today = dates[i]
        yesterday = dates[i - 1]
        
        # ===== DAYセッション =====
        night_start = datetime.combine(yesterday, time(16, 30))
        night_end = datetime.combine(today, time(6, 0))
        prev_night = get_session_ohlc(df_1m, night_start, night_end)
        
        if prev_night:
            day_start = datetime.combine(today, time(8, 45))
            day_end = datetime.combine(today, time(15, 15))
            day_data = get_session(df_1m, day_start, day_end)
            
            if not day_data.empty:
                entry = tick_round(day_data.iloc[0]['Open'])
                gap = abs(entry - prev_night['close']) / prev_night['close']
                
                if gap < GAP_CUT:
                    night_15m = df_15m.loc[night_start:night_end]
                    if len(night_15m) >= 10:
                        slope = calc_slope(night_15m['Close'])
                        action = raptor_signal(prev_night, slope)
                        
                        if action:
                            try:
                                atr = df_daily.loc[:str(yesterday)]['ATR'].iloc[-1]
                                if pd.isna(atr) or atr <= 0:
                                    atr = 400
                            except:
                                atr = 400
                            
                            s_dist = tick_round(atr * stop_mult)
                            t_dist = tick_round(atr * target_mult)
                            
                            if action == 'BUY':
                                stop = entry - s_dist
                                target = entry + t_dist
                            else:
                                stop = entry + s_dist
                                target = entry - t_dist
                            
                            exit_price, reason = execute_trade(day_data, action, entry, stop, target, day_end)
                            diff = (exit_price - entry) if action == 'BUY' else (entry - exit_price)
                            pnl = diff * MULTIPLIER * LOTS - COMMISSION
                            capital += pnl
                            trades.append({'pnl': pnl, 'reason': reason})
        
        # ===== NIGHTセッション =====
        day_start = datetime.combine(today, time(8, 45))
        day_end = datetime.combine(today, time(15, 15))
        prev_day = get_session_ohlc(df_1m, day_start, day_end)
        
        if prev_day:
            night_start = datetime.combine(today, time(16, 30))
            night_end = datetime.combine(today + timedelta(days=1), time(6, 0))
            night_data = get_session(df_1m, night_start, night_end)
            
            if not night_data.empty:
                entry = tick_round(night_data.iloc[0]['Open'])
                gap = abs(entry - prev_day['close']) / prev_day['close']
                
                if gap < GAP_CUT:
                    day_15m = df_15m.loc[day_start:day_end]
                    if len(day_15m) >= 10:
                        slope = calc_slope(day_15m['Close'])
                        action = raptor_signal(prev_day, slope)
                        
                        if action:
                            try:
                                atr = df_daily.loc[:str(today)]['ATR'].iloc[-1]
                                if pd.isna(atr) or atr <= 0:
                                    atr = 400
                            except:
                                atr = 400
                            
                            s_dist = tick_round(atr * stop_mult)
                            t_dist = tick_round(atr * target_mult)
                            
                            if action == 'BUY':
                                stop = entry - s_dist
                                target = entry + t_dist
                            else:
                                stop = entry + s_dist
                                target = entry - t_dist
                            
                            exit_price, reason = execute_trade(night_data, action, entry, stop, target, night_end)
                            diff = (exit_price - entry) if action == 'BUY' else (entry - exit_price)
                            pnl = diff * MULTIPLIER * LOTS - COMMISSION
                            capital += pnl
                            trades.append({'pnl': pnl, 'reason': reason})
    
    # 集計
    if not trades:
        return None
    
    df_t = pd.DataFrame(trades)
    wins = len(df_t[df_t['pnl'] > 0])
    total = len(df_t)
    win_rate = wins / total * 100 if total > 0 else 0
    
    gross_win = df_t[df_t['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_t[df_t['pnl'] <= 0]['pnl'].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else 0
    
    months = len(set([(dates[0] + timedelta(days=i)).strftime('%Y-%m') for i in range(len(dates))]))
    monthly = (capital - CAPITAL) / 96  # 約8年 = 96ヶ月
    
    # 決済理由
    reason_counts = df_t['reason'].value_counts().to_dict()
    
    return {
        'trades': total,
        'win_rate': win_rate,
        'pf': pf,
        'return': (capital - CAPITAL) / CAPITAL * 100,
        'monthly': monthly,
        'target_hits': reason_counts.get('TARGET', 0),
        'stop_hits': reason_counts.get('STOP', 0),
        'close_hits': reason_counts.get('CLOSE', 0)
    }

# ============================================================
# メイン (グリッドサーチ)
# ============================================================
def main():
    df_1m = load_data()
    
    # 15分足
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    
    # 日次ATR
    df_daily = df_1m.resample('D').agg({'High': 'max', 'Low': 'min'}).dropna()
    df_daily['ATR'] = (df_daily['High'] - df_daily['Low']).rolling(14).mean()
    
    dates = sorted(set(df_1m.index.date))
    
    print("\n" + "=" * 90)
    print("🔎 Stop/Target グリッドサーチ (マイクロ1枚)")
    print("=" * 90)
    print(f"{'Stop':>5} | {'Tgt':>5} || {'Trades':>6} | {'Win%':>6} | {'PF':>5} | {'Ret%':>7} | {'月平均':>10} | {'TGT':>4} | {'STP':>4} | {'CLS':>5}")
    print("-" * 90)
    
    results = []
    
    for stop_mult, target_mult in itertools.product(STOP_RANGE, TARGET_RANGE):
        res = run_backtest(df_1m, df_15m, df_daily, dates, stop_mult, target_mult)
        
        if res:
            results.append({
                'stop': stop_mult,
                'target': target_mult,
                **res
            })
            
            print(f"{stop_mult:>5} | {target_mult:>5} || {res['trades']:>6} | {res['win_rate']:>5.1f}% | {res['pf']:>5.2f} | {res['return']:>6.1f}% | ¥{res['monthly']:>9,.0f} | {res['target_hits']:>4} | {res['stop_hits']:>4} | {res['close_hits']:>5}")
    
    print("=" * 90)
    
    # ベスト結果
    if results:
        best = max(results, key=lambda x: x['monthly'])
        print(f"\n🏆 ベスト: Stop={best['stop']} Target={best['target']} → 月平均 ¥{best['monthly']:+,.0f}")

if __name__ == "__main__":
    main()
