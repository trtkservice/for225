#!/usr/bin/env python3
"""
Raptor225 バックテスト (診断版)
================================
あらゆる根本的エラーを検出するための詳細ログ出力
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

# 診断対象日付 (この日付のトレードを詳細出力)
DIAG_DATES = ['2025-12-01', '2025-12-02', '2025-12-04']

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
    
    # データ診断
    print("\n📊 データ診断:")
    print(f"   最初の5行:")
    print(df.head().to_string())
    print(f"\n   最後の5行:")
    print(df.tail().to_string())
    
    return df

# ============================================================
# 診断付きバックテスト
# ============================================================
def backtest_with_diagnosis(df_1m):
    # 15分足
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    
    # 日次ATR
    df_daily = df_1m.resample('D').agg({'High': 'max', 'Low': 'min'}).dropna()
    df_daily['ATR'] = (df_daily['High'] - df_daily['Low']).rolling(14).mean()
    
    dates = sorted(set(df_1m.index.date))
    
    capital = CAPITAL
    trades = []
    
    print("\n" + "=" * 100)
    print("🔍 詳細診断モード")
    print("=" * 100)
    
    for i in range(1, len(dates)):
        today = dates[i]
        yesterday = dates[i - 1]
        today_str = str(today)
        is_diag = today_str in DIAG_DATES
        
        # ===== DAYセッション診断 =====
        night_start = datetime.combine(yesterday, time(16, 30))
        night_end = datetime.combine(today, time(6, 0))
        day_start = datetime.combine(today, time(8, 45))
        day_end = datetime.combine(today, time(15, 15))
        
        night_data = df_1m.loc[night_start:night_end]
        day_data = df_1m.loc[day_start:day_end]
        
        if is_diag:
            print(f"\n{'='*80}")
            print(f"📅 {today} DAYセッション診断")
            print(f"{'='*80}")
            
            print(f"\n【1. セッション時刻】")
            print(f"   前日NIGHT: {night_start} 〜 {night_end}")
            print(f"   当日DAY:   {day_start} 〜 {day_end}")
            
            print(f"\n【2. データ存在確認】")
            print(f"   NIGHT データ件数: {len(night_data)}")
            print(f"   DAY データ件数:   {len(day_data)}")
            
            if not night_data.empty:
                print(f"   NIGHT 最初の足: {night_data.index[0]} OHLC={night_data.iloc[0][['Open','High','Low','Close']].tolist()}")
                print(f"   NIGHT 最後の足: {night_data.index[-1]} OHLC={night_data.iloc[-1][['Open','High','Low','Close']].tolist()}")
            
            if not day_data.empty:
                print(f"   DAY 最初の足:   {day_data.index[0]} OHLC={day_data.iloc[0][['Open','High','Low','Close']].tolist()}")
                print(f"   DAY 最後の足:   {day_data.index[-1]} OHLC={day_data.iloc[-1][['Open','High','Low','Close']].tolist()}")
        
        if night_data.empty or day_data.empty:
            if is_diag:
                print(f"   ⚠️ データ不足でスキップ")
            continue
        
        # 前セッションOHLC
        prev_open = night_data.iloc[0]['Open']
        prev_close = night_data.iloc[-1]['Close']
        prev_high = night_data['High'].max()
        prev_low = night_data['Low'].min()
        
        # エントリー価格
        entry_raw = day_data.iloc[0]['Open']
        entry = tick_round(entry_raw)
        
        if is_diag:
            print(f"\n【3. 前セッション(NIGHT) OHLC】")
            print(f"   Open:  {prev_open}")
            print(f"   High:  {prev_high}")
            print(f"   Low:   {prev_low}")
            print(f"   Close: {prev_close}")
            print(f"   方向:  {'陽線(+1)' if prev_close > prev_open else '陰線(-1)' if prev_close < prev_open else '同値(0)'}")
            
            print(f"\n【4. DAYエントリー】")
            print(f"   DAY始値(生): {entry_raw}")
            print(f"   DAY始値(丸): {entry}")
        
        # ギャップチェック
        gap = abs(entry - prev_close) / prev_close
        
        if is_diag:
            print(f"\n【5. ギャップチェック】")
            print(f"   ギャップ率: {gap*100:.3f}%")
            print(f"   閾値:       {GAP_CUT*100:.3f}%")
            print(f"   判定:       {'PASS' if gap < GAP_CUT else 'FAIL (スキップ)'}")
        
        if gap >= GAP_CUT:
            continue
        
        # モメンタム計算
        night_15m = df_15m.loc[night_start:night_end]
        
        if is_diag:
            print(f"\n【6. モメンタム(15分足)】")
            print(f"   NIGHT 15分足件数: {len(night_15m)}")
            if not night_15m.empty:
                print(f"   最初: {night_15m.index[0]} Close={night_15m.iloc[0]['Close']}")
                print(f"   最後: {night_15m.index[-1]} Close={night_15m.iloc[-1]['Close']}")
        
        if len(night_15m) < 10:
            if is_diag:
                print(f"   ⚠️ 15分足不足でスキップ")
            continue
        
        slope = calc_slope(night_15m['Close'])
        
        if is_diag:
            print(f"   傾き(slope): {slope:.4f}")
            print(f"   方向:        {'正(+1)' if slope > 0 else '負(-1)'}")
        
        # スコア計算
        score_b = 1 if prev_close > prev_open else -1 if prev_close < prev_open else 0
        score_c = 1 if slope > 0 else -1
        total = score_b + score_c
        
        if is_diag:
            print(f"\n【7. シグナル判定】")
            print(f"   B (前セッション方向): {score_b:+d}")
            print(f"   C (モメンタム方向):   {score_c:+d}")
            print(f"   合計スコア:           {total:+d}")
            print(f"   判定: ", end="")
        
        if total >= 2:
            action = 'BUY'
        elif total <= -2:
            action = 'SELL'
        else:
            action = None
        
        if is_diag:
            print(f"{action if action else 'NO-TRADE'}")
        
        if action is None:
            continue
        
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
        
        if is_diag:
            print(f"\n【8. リスク管理】")
            print(f"   ATR:        {atr:.0f}")
            print(f"   Stop距離:   {s_dist} (ATR×{STOP_MULT})")
            print(f"   Target距離: {t_dist} (ATR×{TARGET_MULT})")
            print(f"   Entry:      {entry}")
            print(f"   Stop:       {stop}")
            print(f"   Target:     {target}")
        
        # トレード実行
        session_high = day_data['High'].max()
        session_low = day_data['Low'].min()
        session_close = day_data.iloc[-1]['Close']
        
        exit_price = None
        reason = None
        
        for _, bar in day_data.iterrows():
            if action == 'BUY':
                if bar['Low'] <= stop:
                    exit_price = stop
                    reason = 'STOP'
                    break
                if bar['High'] >= target:
                    exit_price = target
                    reason = 'TARGET'
                    break
            else:
                if bar['High'] >= stop:
                    exit_price = stop
                    reason = 'STOP'
                    break
                if bar['Low'] <= target:
                    exit_price = target
                    reason = 'TARGET'
                    break
        
        if exit_price is None:
            exit_price = tick_round(session_close)
            reason = 'CLOSE'
        
        if is_diag:
            print(f"\n【9. セッション中の値動き】")
            print(f"   Session High:  {session_high}")
            print(f"   Session Low:   {session_low}")
            print(f"   Session Close: {session_close}")
            
            print(f"\n【10. Stop/Target判定】")
            if action == 'BUY':
                print(f"   BUY: Low({session_low}) <= Stop({stop}) ? {'YES' if session_low <= stop else 'NO'}")
                print(f"   BUY: High({session_high}) >= Target({target}) ? {'YES' if session_high >= target else 'NO'}")
            else:
                print(f"   SELL: High({session_high}) >= Stop({stop}) ? {'YES' if session_high >= stop else 'NO'}")
                print(f"   SELL: Low({session_low}) <= Target({target}) ? {'YES' if session_low <= target else 'NO'}")
        
        # 損益計算
        if action == 'BUY':
            diff = exit_price - entry
        else:
            diff = entry - exit_price
        
        pnl = diff * MULTIPLIER * LOTS - COMMISSION
        capital += pnl
        
        if is_diag:
            print(f"\n【11. 決済結果】")
            print(f"   Exit:   {exit_price} ({reason})")
            print(f"   Diff:   {diff:+} ポイント")
            print(f"   PnL:    {diff} × {MULTIPLIER} × {LOTS} - {COMMISSION} = ¥{pnl:+,.0f}")
            
            # 勝敗の妥当性チェック
            print(f"\n【12. 妥当性チェック】")
            if action == 'BUY':
                if diff > 0:
                    print(f"   ✅ BUY で価格上昇 → 利益 (正常)")
                else:
                    print(f"   ✅ BUY で価格下落 → 損失 (正常)")
            else:
                if diff > 0:
                    print(f"   ✅ SELL で価格下落 → 利益 (正常)")
                else:
                    print(f"   ✅ SELL で価格上昇 → 損失 (正常)")
        
        trades.append({'pnl': pnl, 'reason': reason})
    
    # 結果
    print("\n" + "=" * 100)
    print("📊 バックテスト結果")
    print("=" * 100)
    
    if trades:
        df_t = pd.DataFrame(trades)
        wins = len(df_t[df_t['pnl'] > 0])
        total_trades = len(df_t)
        win_rate = wins / total_trades * 100
        total_pnl = df_t['pnl'].sum()
        
        print(f"   トレード数: {total_trades}")
        print(f"   勝率:       {win_rate:.1f}%")
        print(f"   純損益:     ¥{total_pnl:+,.0f}")
        print(f"   月平均:     ¥{total_pnl/96:+,.0f}")
        
        reason_counts = df_t['reason'].value_counts()
        print(f"\n   決済理由:")
        for r, c in reason_counts.items():
            print(f"     {r}: {c}回 ({c/total_trades*100:.1f}%)")

if __name__ == "__main__":
    df = load_data()
    backtest_with_diagnosis(df)
