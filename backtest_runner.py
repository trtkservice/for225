#!/usr/bin/env python3
"""
Raptor225 バックテスト (バグ修正版)
====================================
楽天証券 / 資金10万円 / 日経225マイクロ1枚 / デイトレード

修正: NIGHTセッションの「前セッション」参照を正しく実装
- DAY: 前のカレンダー日のNIGHT (昨日16:30〜今日06:00)
- NIGHT: 同日のDAY (今日08:45〜今日15:15)
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from datetime import datetime, time, timedelta
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 設定
# ============================================================
CAPITAL = 100_000
LOTS = 1
MULTIPLIER = 10
COMMISSION = 22
TICK = 5
GAP_CUT = 0.0025
STOP_MULT = 1.0
TARGET_MULT = 2.0

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
# セッション処理
# ============================================================
def get_session_data(df_1m, start_dt, end_dt):
    """指定時間範囲のデータを取得"""
    data = df_1m.loc[start_dt:end_dt]
    return data if not data.empty else None

def get_ohlc(data):
    """データからOHLC取得"""
    if data is None or data.empty:
        return None
    return {
        'open': data.iloc[0]['Open'],
        'high': data['High'].max(),
        'low': data['Low'].min(),
        'close': data.iloc[-1]['Close']
    }

def get_signal(prev_ohlc, slope):
    """Raptorシグナル生成"""
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

def execute_trade(session_data, action, entry, stop, target):
    """トレード実行"""
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
# バックテスト
# ============================================================
def backtest(df_1m):
    # 15分足
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    
    # 日次ATR
    df_daily = df_1m.resample('D').agg({'High': 'max', 'Low': 'min'}).dropna()
    df_daily['ATR'] = (df_daily['High'] - df_daily['Low']).rolling(14).mean()
    
    # 取引日リスト
    dates = sorted(set(df_1m.index.date))
    
    capital = CAPITAL
    trades = []
    day_trades = 0
    night_trades = 0
    
    print(f"\n🚀 バックテスト開始 ({len(dates)}日)")
    print(f"   マイクロ{LOTS}枚, Stop {STOP_MULT} ATR, Target {TARGET_MULT} ATR")
    print("-" * 70)
    
    for today in dates:
        # ================================================================
        # DAYセッション (08:45〜15:15)
        # 直前セッション = 前のカレンダー日のNIGHT (昨日16:30〜今日06:00)
        # ================================================================
        yesterday_cal = today - timedelta(days=1)  # カレンダー上の昨日
        
        prev_night_start = datetime.combine(yesterday_cal, time(16, 30))
        prev_night_end = datetime.combine(today, time(6, 0))
        day_start = datetime.combine(today, time(8, 45))
        day_end = datetime.combine(today, time(15, 15))
        
        prev_night_data = get_session_data(df_1m, prev_night_start, prev_night_end)
        day_data = get_session_data(df_1m, day_start, day_end)
        
        if prev_night_data is not None and day_data is not None and len(prev_night_data) >= 100:
            prev_ohlc = get_ohlc(prev_night_data)
            entry = tick_round(day_data.iloc[0]['Open'])
            
            # ギャップチェック
            gap = abs(entry - prev_ohlc['close']) / prev_ohlc['close']
            if gap < GAP_CUT:
                # モメンタム
                prev_15m = df_15m.loc[prev_night_start:prev_night_end]
                if len(prev_15m) >= 10:
                    slope = calc_slope(prev_15m['Close'])
                    action = get_signal(prev_ohlc, slope)
                    
                    if action:
                        # ATR
                        try:
                            atr = df_daily.loc[:str(yesterday_cal)]['ATR'].iloc[-1]
                            if pd.isna(atr) or atr <= 0:
                                atr = 400
                        except:
                            atr = 400
                        
                        s_dist = tick_round(atr * STOP_MULT)
                        t_dist = tick_round(atr * TARGET_MULT)
                        
                        if action == 'BUY':
                            stop = entry - s_dist
                            target = entry + t_dist
                        else:
                            stop = entry + s_dist
                            target = entry - t_dist
                        
                        exit_price, reason = execute_trade(day_data, action, entry, stop, target)
                        diff = (exit_price - entry) if action == 'BUY' else (entry - exit_price)
                        pnl = diff * MULTIPLIER * LOTS - COMMISSION
                        capital += pnl
                        
                        trades.append({
                            'date': today, 'session': 'DAY', 'action': action,
                            'pnl': pnl, 'reason': reason
                        })
                        day_trades += 1
                        
                        if len(trades) <= 5:
                            print(f"  #{len(trades)} {today} DAY {action} {entry}→{exit_price} PnL={pnl:+,.0f}")
        
        # ================================================================
        # NIGHTセッション (16:30〜翌06:00)
        # 直前セッション = 同日のDAY (今日08:45〜今日15:15)
        # ================================================================
        prev_day_start = datetime.combine(today, time(8, 45))
        prev_day_end = datetime.combine(today, time(15, 15))
        night_start = datetime.combine(today, time(16, 30))
        night_end = datetime.combine(today + timedelta(days=1), time(6, 0))
        
        prev_day_data = get_session_data(df_1m, prev_day_start, prev_day_end)
        night_data = get_session_data(df_1m, night_start, night_end)
        
        if prev_day_data is not None and night_data is not None and len(prev_day_data) >= 100:
            prev_ohlc = get_ohlc(prev_day_data)
            entry = tick_round(night_data.iloc[0]['Open'])
            
            # ギャップチェック
            gap = abs(entry - prev_ohlc['close']) / prev_ohlc['close']
            if gap < GAP_CUT:
                # モメンタム
                prev_15m = df_15m.loc[prev_day_start:prev_day_end]
                if len(prev_15m) >= 10:
                    slope = calc_slope(prev_15m['Close'])
                    action = get_signal(prev_ohlc, slope)
                    
                    if action:
                        # ATR
                        try:
                            atr = df_daily.loc[:str(today)]['ATR'].iloc[-1]
                            if pd.isna(atr) or atr <= 0:
                                atr = 400
                        except:
                            atr = 400
                        
                        s_dist = tick_round(atr * STOP_MULT)
                        t_dist = tick_round(atr * TARGET_MULT)
                        
                        if action == 'BUY':
                            stop = entry - s_dist
                            target = entry + t_dist
                        else:
                            stop = entry + s_dist
                            target = entry - t_dist
                        
                        exit_price, reason = execute_trade(night_data, action, entry, stop, target)
                        diff = (exit_price - entry) if action == 'BUY' else (entry - exit_price)
                        pnl = diff * MULTIPLIER * LOTS - COMMISSION
                        capital += pnl
                        
                        trades.append({
                            'date': today, 'session': 'NIGHT', 'action': action,
                            'pnl': pnl, 'reason': reason
                        })
                        night_trades += 1
                        
                        if len(trades) <= 5:
                            print(f"  #{len(trades)} {today} NIGHT {action} {entry}→{exit_price} PnL={pnl:+,.0f}")
    
    # 結果
    print("-" * 70)
    
    if not trades:
        print("⚠️ トレードなし")
        return
    
    df_t = pd.DataFrame(trades)
    wins = len(df_t[df_t['pnl'] > 0])
    total = len(df_t)
    win_rate = wins / total * 100
    
    gross_win = df_t[df_t['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_t[df_t['pnl'] <= 0]['pnl'].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    
    total_pnl = df_t['pnl'].sum()
    monthly = total_pnl / 96  # 約8年
    
    print(f"\n📊 結果")
    print("=" * 70)
    print(f"  期間        : {df_t['date'].min()} 〜 {df_t['date'].max()}")
    print(f"  トレード数  : {total}回 (DAY:{day_trades} NIGHT:{night_trades})")
    print(f"  勝率        : {win_rate:.1f}% ({wins}勝 {total-wins}敗)")
    print(f"  PF          : {pf:.2f}")
    print("=" * 70)
    print(f"  最終資金    : ¥{capital:,.0f}")
    print(f"  純損益      : ¥{total_pnl:+,.0f}")
    print(f"  リターン    : {(capital - CAPITAL) / CAPITAL * 100:+.1f}%")
    print(f"  月平均      : ¥{monthly:+,.0f}")
    print("=" * 70)
    
    reason_counts = df_t['reason'].value_counts()
    print(f"\n📈 決済理由:")
    for r, c in reason_counts.items():
        print(f"  {r}: {c}回 ({c/total*100:.1f}%)")

if __name__ == "__main__":
    df = load_data()
    backtest(df)
