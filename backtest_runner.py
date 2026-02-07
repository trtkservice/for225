#!/usr/bin/env python3
"""
Raptor225 Backtest - 日足版
============================
楽天証券 / 資金10万円 / 日経225マイクロ1枚

データ: 日足OHLC (始値・高値・安値・終値)
判定ロジック: Raptor (B+C)
グリッドテスト: Stop/Target × デイトレ/オーバーナイト
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from datetime import datetime
from itertools import product
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 設定
# ============================================================
CAPITAL = 100_000      # 資金10万円
LOTS = 1               # マイクロ1枚
MULTIPLIER = 10        # 1ポイント = 10円
COMMISSION = 22        # 往復手数料
TICK = 5               # 呼値

# グリッドサーチ範囲
STOP_RANGE = [0.3, 0.5, 0.7, 1.0]
TARGET_RANGE = [0.5, 1.0, 1.5, 2.0]
MODE_RANGE = ['DAY', 'OVERNIGHT']  # デイトレ or オーバーナイト

# モメンタム計算期間
MOMENTUM_PERIOD = 5  # 直近5日


# ============================================================
# ユーティリティ
# ============================================================
def tick_round(price):
    return int(round(price / TICK) * TICK)


def calc_slope(closes):
    """終値配列の回帰傾き"""
    n = len(closes)
    if n < 2:
        return 0
    x = np.arange(n)
    return float(np.polyfit(x, closes.values, 1)[0])


# ============================================================
# データ読み込み (日足に変換)
# ============================================================
def load_daily_data():
    """N225minif_*.xlsx を読み込み、日足OHLCに変換"""
    base = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(base, "N225minif_*.xlsx")))
    
    if not files:
        print("❌ N225minif_*.xlsx が見つかりません")
        sys.exit(1)
    
    print(f"📥 {len(files)}ファイル読み込み中...")
    dfs = [pd.read_excel(f) for f in files]
    
    df = pd.concat(dfs, ignore_index=True)
    df.rename(columns={
        '日付': 'Date', '時間': 'Time', '時刻': 'Time',
        '始値': 'Open', '高値': 'High', '安値': 'Low', '終値': 'Close'
    }, inplace=True)
    
    df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df = df.set_index('Datetime').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df = df[['Open', 'High', 'Low', 'Close']].astype(float)
    
    # 日足に変換
    daily = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()
    
    # 直近3年に絞る
    daily = daily[daily.index >= '2023-01-01']
    
    print(f"✅ 日足 {len(daily)}本 ({daily.index[0].date()} 〜 {daily.index[-1].date()})")
    return daily


# ============================================================
# Raptorシグナル判定
# ============================================================
def get_raptor_signal(daily, idx):
    """
    Raptorシグナル判定
    
    B: 前日の方向 (陽線+1, 陰線-1)
    C: 直近N日のモメンタム (正+1, 負-1)
    
    B + C >= +2 → BUY
    B + C <= -2 → SELL
    """
    if idx < MOMENTUM_PERIOD:
        return None
    
    # 前日のデータ
    prev = daily.iloc[idx - 1]
    
    # B: 前日の方向
    if prev['Close'] > prev['Open']:
        score_b = 1
    elif prev['Close'] < prev['Open']:
        score_b = -1
    else:
        score_b = 0
    
    # C: モメンタム (直近N日の傾き)
    closes = daily['Close'].iloc[idx - MOMENTUM_PERIOD:idx]
    slope = calc_slope(closes)
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
def execute_day_trade(daily, idx, action, stop_mult, target_mult):
    """
    デイトレード: 当日始値でエントリー、当日中にStop/Target判定、当日終値で決済
    """
    today = daily.iloc[idx]
    prev = daily.iloc[idx - 1]
    
    entry = tick_round(today['Open'])
    
    # ATR (前日のレンジで代用)
    atr = prev['High'] - prev['Low']
    if atr <= 0:
        atr = 300
    
    s_dist = tick_round(atr * stop_mult)
    t_dist = tick_round(atr * target_mult)
    
    if action == 'BUY':
        stop = entry - s_dist
        target = entry + t_dist
        
        # 当日の高値・安値でStop/Target判定
        if today['Low'] <= stop:
            exit_price = stop
            reason = 'STOP'
        elif today['High'] >= target:
            exit_price = target
            reason = 'TARGET'
        else:
            exit_price = tick_round(today['Close'])
            reason = 'CLOSE'
        
        diff = exit_price - entry
    else:  # SELL
        stop = entry + s_dist
        target = entry - t_dist
        
        if today['High'] >= stop:
            exit_price = stop
            reason = 'STOP'
        elif today['Low'] <= target:
            exit_price = target
            reason = 'TARGET'
        else:
            exit_price = tick_round(today['Close'])
            reason = 'CLOSE'
        
        diff = entry - exit_price
    
    pnl = diff * MULTIPLIER * LOTS - COMMISSION
    
    return {
        'date': daily.index[idx].date(),
        'action': action,
        'entry': entry,
        'exit': exit_price,
        'stop': stop,
        'target': target,
        'reason': reason,
        'pnl': pnl,
        'atr': atr
    }


def execute_overnight_trade(daily, idx, action, stop_mult, target_mult):
    """
    オーバーナイト: 当日始値でエントリー、翌日終値で決済 (途中でStop/Targetあり)
    """
    if idx + 1 >= len(daily):
        return None
    
    today = daily.iloc[idx]
    tomorrow = daily.iloc[idx + 1]
    prev = daily.iloc[idx - 1]
    
    entry = tick_round(today['Open'])
    
    # ATR (前日のレンジで代用)
    atr = prev['High'] - prev['Low']
    if atr <= 0:
        atr = 300
    
    s_dist = tick_round(atr * stop_mult)
    t_dist = tick_round(atr * target_mult)
    
    if action == 'BUY':
        stop = entry - s_dist
        target = entry + t_dist
        
        # 当日判定
        if today['Low'] <= stop:
            exit_price = stop
            reason = 'STOP_D1'
        elif today['High'] >= target:
            exit_price = target
            reason = 'TARGET_D1'
        # 翌日判定
        elif tomorrow['Low'] <= stop:
            exit_price = stop
            reason = 'STOP_D2'
        elif tomorrow['High'] >= target:
            exit_price = target
            reason = 'TARGET_D2'
        else:
            exit_price = tick_round(tomorrow['Close'])
            reason = 'CLOSE_D2'
        
        diff = exit_price - entry
    else:  # SELL
        stop = entry + s_dist
        target = entry - t_dist
        
        if today['High'] >= stop:
            exit_price = stop
            reason = 'STOP_D1'
        elif today['Low'] <= target:
            exit_price = target
            reason = 'TARGET_D1'
        elif tomorrow['High'] >= stop:
            exit_price = stop
            reason = 'STOP_D2'
        elif tomorrow['Low'] <= target:
            exit_price = target
            reason = 'TARGET_D2'
        else:
            exit_price = tick_round(tomorrow['Close'])
            reason = 'CLOSE_D2'
        
        diff = entry - exit_price
    
    pnl = diff * MULTIPLIER * LOTS - COMMISSION
    
    return {
        'date': daily.index[idx].date(),
        'action': action,
        'entry': entry,
        'exit': exit_price,
        'stop': stop,
        'target': target,
        'reason': reason,
        'pnl': pnl,
        'atr': atr
    }


# ============================================================
# バックテスト実行
# ============================================================
def run_backtest(daily, stop_mult, target_mult, mode):
    """1パラメータセットでバックテスト"""
    trades = []
    
    for i in range(MOMENTUM_PERIOD, len(daily)):
        action = get_raptor_signal(daily, i)
        if action is None:
            continue
        
        if mode == 'DAY':
            trade = execute_day_trade(daily, i, action, stop_mult, target_mult)
        else:  # OVERNIGHT
            trade = execute_overnight_trade(daily, i, action, stop_mult, target_mult)
        
        if trade:
            trades.append(trade)
    
    if not trades:
        return None
    
    df = pd.DataFrame(trades)
    wins = len(df[df['pnl'] > 0])
    total = len(df)
    total_pnl = df['pnl'].sum()
    monthly = total_pnl / 36  # 3年 = 36ヶ月
    
    return {
        'trades': total,
        'win_rate': wins / total * 100,
        'total_pnl': total_pnl,
        'monthly': monthly,
        'details': df
    }


# ============================================================
# メイン
# ============================================================
def main():
    daily = load_daily_data()
    
    print("\n" + "=" * 100)
    print("🔎 グリッドサーチ: Stop × Target × Mode")
    print("=" * 100)
    print(f"{'Mode':<10} | {'Stop':>5} | {'Tgt':>5} || {'Trades':>6} | {'Win%':>6} | {'総損益':>12} | {'月平均':>10}")
    print("-" * 100)
    
    results = []
    
    for mode, stop_mult, target_mult in product(MODE_RANGE, STOP_RANGE, TARGET_RANGE):
        res = run_backtest(daily, stop_mult, target_mult, mode)
        
        if res:
            results.append({
                'mode': mode,
                'stop': stop_mult,
                'target': target_mult,
                **res
            })
            
            print(f"{mode:<10} | {stop_mult:>5} | {target_mult:>5} || {res['trades']:>6} | "
                  f"{res['win_rate']:>5.1f}% | ¥{res['total_pnl']:>+10,.0f} | ¥{res['monthly']:>+8,.0f}")
    
    print("=" * 100)
    
    # ベスト結果
    if results:
        best = max(results, key=lambda x: x['monthly'])
        print(f"\n🏆 ベスト: {best['mode']} Stop={best['stop']} Target={best['target']}")
        print(f"   月平均: ¥{best['monthly']:+,.0f} | 総損益: ¥{best['total_pnl']:+,.0f}")
        
        # 2025年12月の詳細ログ
        print(f"\n{'='*100}")
        print(f"📅 2025年12月 詳細ログ ({best['mode']} Stop={best['stop']} Target={best['target']})")
        print("=" * 100)
        
        dec_trades = best['details'][
            best['details']['date'].apply(lambda d: d.year == 2025 and d.month == 12)
        ]
        
        if len(dec_trades) > 0:
            for _, t in dec_trades.iterrows():
                print(f"  {t['date']} | {t['action']:4} | Entry:{t['entry']:,.0f} → Exit:{t['exit']:,.0f} | "
                      f"Stop:{t['stop']:,.0f} Target:{t['target']:,.0f} | {t['reason']:<10} | "
                      f"ATR:{t['atr']:.0f} | PnL:¥{t['pnl']:+,.0f}")
        else:
            print("  (2025年12月のトレードなし)")


if __name__ == "__main__":
    main()
