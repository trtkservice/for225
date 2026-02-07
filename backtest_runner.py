#!/usr/bin/env python3
"""
Raptor225 バックテスト
======================
楽天証券 / 資金10万円 / 日経225マイクロ1枚 / デイトレード

セッション:
  - DAY:   08:45〜15:15 (判定08:00、直前=前日NIGHT)
  - NIGHT: 16:30〜翌06:00 (判定16:00、直前=同日DAY)
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
STOP_MULT = 1.0        # ストップ = 1.0 ATR
TARGET_MULT = 2.0      # ターゲット = 2.0 ATR

# ============================================================
# ユーティリティ
# ============================================================
def tick_round(price):
    """5円刻みに丸める"""
    return int(round(price / TICK) * TICK)

def calc_slope(closes):
    """終値配列の回帰傾き"""
    n = len(closes)
    if n < 2:
        return 0
    x = np.arange(n)
    y = closes.values if hasattr(closes, 'values') else closes
    slope = np.polyfit(x, y, 1)[0]
    return slope

# ============================================================
# データ読み込み
# ============================================================
def load_data():
    """N225minif_*.xlsx を読み込み"""
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
    
    # カラム名統一
    df.rename(columns={
        '日付': 'Date', '時間': 'Time', '時刻': 'Time',
        '始値': 'Open', '高値': 'High', '安値': 'Low', '終値': 'Close'
    }, inplace=True)
    
    # Datetime化
    df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df = df.set_index('Datetime').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df = df[['Open', 'High', 'Low', 'Close']].astype(float)
    
    print(f"✅ {len(df):,}本 ({df.index[0]} 〜 {df.index[-1]})")
    return df

# ============================================================
# セッションデータ抽出
# ============================================================
def get_session(df_1m, start_dt, end_dt):
    """指定時間範囲の1分足を取得"""
    return df_1m.loc[start_dt:end_dt]

def get_session_ohlc(df_1m, start_dt, end_dt):
    """セッションのOHLC (始値, 高値, 安値, 終値)"""
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
# Raptorシグナル判定
# ============================================================
def raptor_signal(prev_ohlc, slope):
    """
    B: 直前セッションの方向 (陽線+1, 陰線-1)
    C: モメンタム傾き (正+1, 負-1)
    合計 >= +2: BUY, <= -2: SELL, それ以外: NO-TRADE
    """
    # B判定
    if prev_ohlc['close'] > prev_ohlc['open']:
        score_b = 1
    elif prev_ohlc['close'] < prev_ohlc['open']:
        score_b = -1
    else:
        score_b = 0
    
    # C判定
    score_c = 1 if slope > 0 else -1
    
    total = score_b + score_c
    
    if total >= 2:
        return 'BUY'
    elif total <= -2:
        return 'SELL'
    else:
        return None

# ============================================================
# トレード実行
# ============================================================
def execute_trade(df_1m, action, entry_price, stop, target, session_end):
    """
    分足をループしてStop/Target判定、ヒットしなければセッション終了時決済
    """
    session_data = df_1m.loc[:session_end]
    
    for _, bar in session_data.iterrows():
        if action == 'BUY':
            if bar['Low'] <= stop:
                return stop, 'STOP'
            if bar['High'] >= target:
                return target, 'TARGET'
        else:  # SELL
            if bar['High'] >= stop:
                return stop, 'STOP'
            if bar['Low'] <= target:
                return target, 'TARGET'
    
    # セッション終了時決済
    if not session_data.empty:
        return tick_round(session_data.iloc[-1]['Close']), 'CLOSE'
    return entry_price, 'NO_DATA'

# ============================================================
# バックテスト本体
# ============================================================
def backtest(df_1m):
    # 15分足作成
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    
    # 日次ATR (14日平均レンジ)
    df_daily = df_1m.resample('D').agg({
        'High': 'max', 'Low': 'min'
    }).dropna()
    df_daily['ATR'] = (df_daily['High'] - df_daily['Low']).rolling(14).mean()
    
    # 取引日リスト
    dates = sorted(set(df_1m.index.date))
    
    capital = CAPITAL
    trades = []
    
    print(f"\n🚀 バックテスト開始 ({len(dates)}日)")
    print(f"   マイクロ{LOTS}枚, Stop {STOP_MULT} ATR, Target {TARGET_MULT} ATR")
    print("-" * 60)
    
    for i in range(1, len(dates)):
        today = dates[i]
        yesterday = dates[i - 1]
        
        # ========== DAYセッション ==========
        # 直前 = 前日NIGHT (昨日16:30〜今日06:00)
        night_start = datetime.combine(yesterday, time(16, 30))
        night_end = datetime.combine(today, time(6, 0))
        prev_night = get_session_ohlc(df_1m, night_start, night_end)
        
        if prev_night:
            # DAYセッション時間
            day_start = datetime.combine(today, time(8, 45))
            day_end = datetime.combine(today, time(15, 15))
            day_data = get_session(df_1m, day_start, day_end)
            
            if not day_data.empty:
                entry = tick_round(day_data.iloc[0]['Open'])
                
                # ギャップチェック
                gap = abs(entry - prev_night['close']) / prev_night['close']
                
                if gap < GAP_CUT:
                    # モメンタム (直前NIGHTの15分足)
                    night_15m = df_15m.loc[night_start:night_end]
                    if len(night_15m) >= 10:
                        slope = calc_slope(night_15m['Close'])
                        action = raptor_signal(prev_night, slope)
                        
                        if action:
                            # ATR取得
                            try:
                                atr = df_daily.loc[:str(yesterday)]['ATR'].iloc[-1]
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
                            
                            # トレード実行
                            exit_price, reason = execute_trade(
                                day_data, action, entry, stop, target, day_end
                            )
                            
                            diff = (exit_price - entry) if action == 'BUY' else (entry - exit_price)
                            pnl = diff * MULTIPLIER * LOTS - COMMISSION
                            capital += pnl
                            
                            trades.append({
                                'date': today, 'session': 'DAY', 'action': action,
                                'entry': entry, 'exit': exit_price, 'pnl': pnl,
                                'reason': reason
                            })
                            
                            if len(trades) <= 5:
                                print(f"  #{len(trades)} {today} DAY {action} {entry}→{exit_price} PnL={pnl:+,.0f}")
                            
                            # 2025年12月の詳細ログ
                            if today.year == 2025 and today.month == 12:
                                print(f"  [DEC] {today} DAY | {action} | Entry:{entry} Stop:{stop} Target:{target} | ATR:{atr:.0f} | Exit:{exit_price}({reason}) | PnL:{pnl:+,.0f}")
        
        # ========== NIGHTセッション ==========
        # 直前 = 同日DAY (今日08:45〜15:15)
        day_start = datetime.combine(today, time(8, 45))
        day_end = datetime.combine(today, time(15, 15))
        prev_day = get_session_ohlc(df_1m, day_start, day_end)
        
        if prev_day:
            # NIGHTセッション時間
            night_start = datetime.combine(today, time(16, 30))
            night_end = datetime.combine(today + timedelta(days=1), time(6, 0))
            night_data = get_session(df_1m, night_start, night_end)
            
            if not night_data.empty:
                entry = tick_round(night_data.iloc[0]['Open'])
                
                # ギャップチェック
                gap = abs(entry - prev_day['close']) / prev_day['close']
                
                if gap < GAP_CUT:
                    # モメンタム (直前DAYの15分足)
                    day_15m = df_15m.loc[day_start:day_end]
                    if len(day_15m) >= 10:
                        slope = calc_slope(day_15m['Close'])
                        action = raptor_signal(prev_day, slope)
                        
                        if action:
                            # ATR取得
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
                            
                            # トレード実行
                            exit_price, reason = execute_trade(
                                night_data, action, entry, stop, target, night_end
                            )
                            
                            diff = (exit_price - entry) if action == 'BUY' else (entry - exit_price)
                            pnl = diff * MULTIPLIER * LOTS - COMMISSION
                            capital += pnl
                            
                            trades.append({
                                'date': today, 'session': 'NIGHT', 'action': action,
                                'entry': entry, 'exit': exit_price, 'pnl': pnl,
                                'reason': reason
                            })
                            
                            if len(trades) <= 5:
                                print(f"  #{len(trades)} {today} NIGHT {action} {entry}→{exit_price} PnL={pnl:+,.0f}")
                            
                            # 2025年12月の詳細ログ
                            if today.year == 2025 and today.month == 12:
                                print(f"  [DEC] {today} NIGHT | {action} | Entry:{entry} Stop:{stop} Target:{target} | ATR:{atr:.0f} | Exit:{exit_price}({reason}) | PnL:{pnl:+,.0f}")
    
    # ========== 結果集計 ==========
    print("-" * 60)
    
    if not trades:
        print("⚠️ トレードなし")
        return
    
    df_t = pd.DataFrame(trades)
    
    wins = df_t[df_t['pnl'] > 0]
    losses = df_t[df_t['pnl'] <= 0]
    
    total_pnl = df_t['pnl'].sum()
    win_rate = len(wins) / len(df_t) * 100
    
    gross_win = wins['pnl'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    
    months = len(set([t['date'].strftime('%Y-%m') for t in trades]))
    monthly = total_pnl / months if months > 0 else 0
    
    day_t = df_t[df_t['session'] == 'DAY']
    night_t = df_t[df_t['session'] == 'NIGHT']
    
    print(f"\n📊 結果")
    print("=" * 60)
    print(f"  期間      : {df_t['date'].min()} 〜 {df_t['date'].max()}")
    print(f"  トレード  : {len(df_t)}回 (DAY:{len(day_t)} NIGHT:{len(night_t)})")
    print(f"  勝率      : {win_rate:.1f}% ({len(wins)}勝 {len(losses)}敗)")
    print(f"  PF        : {pf:.2f}")
    print("=" * 60)
    print(f"  最終資金  : ¥{capital:,.0f}")
    print(f"  純損益    : ¥{total_pnl:+,.0f}")
    print(f"  リターン  : {(capital - CAPITAL) / CAPITAL * 100:+.1f}%")
    print(f"  月平均    : ¥{monthly:+,.0f}")
    print("=" * 60)
    
    # 決済理由の内訳
    if 'reason' in df_t.columns:
        reason_counts = df_t['reason'].value_counts()
        print("\n📈 決済理由内訳:")
        for r, cnt in reason_counts.items():
            pct = cnt / len(df_t) * 100
            print(f"  {r}: {cnt}回 ({pct:.1f}%)")

# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    df = load_data()
    backtest(df)
