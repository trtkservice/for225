#!/usr/bin/env python3
"""
Raptor225 Backtest - 内部プロンプト完全準拠版
=============================================
Raptorの判定ロジックを正確に再現

判定ロジック:
- B: 直前セッションの実体方向 (陽線+1, 陰線-1, 同値0)
- C: 直近15M 32本の回帰傾き (正+1, 負-1, 微小0)
- D: 過熱判定 (直前レンジ >= 直近10セッション平均×1.8 なら B→0へ弱める)
- TotalScore = B + C
- TotalScore >= +2 → BUY, <= -2 → SELL, else → NO-TRADE

セッション:
- DAY: 08:45開始, 15:45終了
- NIGHT: 16:30開始, 06:00終了
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from datetime import datetime, time, timedelta
from itertools import product
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# Raptorパラメータ (内部プロンプト準拠)
# ============================================================
G_CUT = 0.0025           # ギャップ閾値 0.25%
N_MOMENTUM = 32          # モメンタム計算に使う15分足の本数
R_OVERHEAT = 1.8         # 過熱判定倍率
SLOPE_THRESHOLD = 0.5    # 傾きがこれ以下なら C=0

# バックテスト設定
CAPITAL = 100_000
LOTS = 1
MULTIPLIER = 10
COMMISSION = 0           # Raptor準拠: cost=0
TICK = 5

# セッション時刻 (Raptor準拠)
DAY_OPEN = time(8, 45)
DAY_CLOSE = time(15, 45)  # 15:45
NIGHT_OPEN = time(16, 30)
NIGHT_CLOSE = time(6, 0)

# グリッドサーチ範囲
STOP_RANGE = [0.5, 0.7, 1.0]
TARGET_RANGE = [0.5, 1.0, 1.5]


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
# データ読み込み
# ============================================================
def load_data():
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
    
    # 直近3年に絞る
    df = df[df.index >= '2023-01-01']
    
    print(f"✅ {len(df):,}本 (2023年以降)")
    return df


# ============================================================
# Raptorロジック (内部プロンプト完全準拠)
# ============================================================
def get_raptor_signal(
    prev_ohlc: dict,
    prev_15m_closes: pd.Series,
    recent_ranges: pd.Series
) -> dict:
    """
    Raptorシグナル判定
    
    Args:
        prev_ohlc: 直前セッションのOHLC
        prev_15m_closes: 直前セッションの15分足終値 (最後N_MOMENTUM本)
        recent_ranges: 直近10セッションのレンジ
    
    Returns:
        dict with signal details
    """
    result = {
        'action': None,
        'score_b': 0,
        'score_b_adj': 0,
        'score_c': 0,
        'total': 0,
        'is_overheat': False
    }
    
    # B: 直前セッション方向
    if prev_ohlc['close'] > prev_ohlc['open']:
        score_b = 1
    elif prev_ohlc['close'] < prev_ohlc['open']:
        score_b = -1
    else:
        score_b = 0
    result['score_b'] = score_b
    
    # C: モメンタム (15分足 N本の傾き)
    if len(prev_15m_closes) >= N_MOMENTUM:
        slope = calc_slope(prev_15m_closes.iloc[-N_MOMENTUM:])
        if abs(slope) < SLOPE_THRESHOLD:
            score_c = 0  # 微小
        else:
            score_c = 1 if slope > 0 else -1
    else:
        score_c = 0
    result['score_c'] = score_c
    
    # D: 過熱判定
    prev_range = prev_ohlc['high'] - prev_ohlc['low']
    avg_range = recent_ranges.mean() if len(recent_ranges) > 0 else 500
    
    if avg_range > 0 and prev_range >= avg_range * R_OVERHEAT:
        score_b_adj = 0  # Bを0へ弱める
        result['is_overheat'] = True
    else:
        score_b_adj = score_b
    result['score_b_adj'] = score_b_adj
    
    # TotalScore
    total = score_b_adj + score_c
    result['total'] = total
    
    # 判定
    if total >= 2:
        result['action'] = 'BUY'
    elif total <= -2:
        result['action'] = 'SELL'
    else:
        result['action'] = None  # NO-TRADE
    
    return result


# ============================================================
# セッションデータ取得
# ============================================================
def get_session_data(df_1m, session_type, date):
    """セッション時間帯のデータを取得"""
    if session_type == 'DAY':
        start = datetime.combine(date, DAY_OPEN)
        end = datetime.combine(date, DAY_CLOSE)
    else:  # NIGHT
        start = datetime.combine(date, NIGHT_OPEN)
        end = datetime.combine(date + timedelta(days=1), NIGHT_CLOSE)
    
    data = df_1m.loc[start:end]
    if data.empty:
        return None
    
    return {
        'data': data,
        'open': data.iloc[0]['Open'],
        'high': data['High'].max(),
        'low': data['Low'].min(),
        'close': data.iloc[-1]['Close']
    }


def get_prev_session_data(df_1m, session_type, date):
    """
    直前セッションのデータを取得
    - DAY: 前日のNIGHT (前日16:30〜当日06:00)
    - NIGHT: 同日のDAY (当日08:45〜15:45)
    """
    if session_type == 'DAY':
        # 前日のNIGHT
        prev_date = date - timedelta(days=1)
        start = datetime.combine(prev_date, NIGHT_OPEN)
        end = datetime.combine(date, NIGHT_CLOSE)
    else:  # NIGHT
        # 同日のDAY
        start = datetime.combine(date, DAY_OPEN)
        end = datetime.combine(date, DAY_CLOSE)
    
    data = df_1m.loc[start:end]
    if data.empty or len(data) < 100:
        return None
    
    return {
        'data': data,
        'open': data.iloc[0]['Open'],
        'high': data['High'].max(),
        'low': data['Low'].min(),
        'close': data.iloc[-1]['Close']
    }


# ============================================================
# トレード実行
# ============================================================
def execute_trade(session_data, action, entry, stop_mult, target_mult, prev_range):
    """トレード実行 (寄り引け + Stop/Target)"""
    data = session_data['data']
    
    # ATR = 前セッションのレンジ
    atr = prev_range if prev_range > 0 else 300
    
    s_dist = tick_round(atr * stop_mult)
    t_dist = tick_round(atr * target_mult)
    
    if action == 'BUY':
        stop = entry - s_dist
        target = entry + t_dist
        
        # 高値・安値でStop/Target判定
        if session_data['low'] <= stop:
            exit_price = stop
            reason = 'STOP'
        elif session_data['high'] >= target:
            exit_price = target
            reason = 'TARGET'
        else:
            exit_price = tick_round(session_data['close'])
            reason = 'CLOSE'
        
        diff = exit_price - entry
    else:  # SELL
        stop = entry + s_dist
        target = entry - t_dist
        
        if session_data['high'] >= stop:
            exit_price = stop
            reason = 'STOP'
        elif session_data['low'] <= target:
            exit_price = target
            reason = 'TARGET'
        else:
            exit_price = tick_round(session_data['close'])
            reason = 'CLOSE'
        
        diff = entry - exit_price
    
    pnl = diff * MULTIPLIER * LOTS - COMMISSION
    
    return {
        'entry': entry,
        'exit': exit_price,
        'stop': stop,
        'target': target,
        'reason': reason,
        'pnl': pnl,
        'atr': atr
    }


# ============================================================
# バックテスト本体
# ============================================================
def run_backtest(df_1m, df_15m, stop_mult, target_mult):
    """1パラメータセットでバックテスト"""
    dates = sorted(set(df_1m.index.date))
    trades = []
    
    # セッション履歴 (過熱判定用)
    session_ranges = []
    
    for date in dates:
        for session_type in ['DAY', 'NIGHT']:
            # 直前セッション取得
            prev = get_prev_session_data(df_1m, session_type, date)
            if prev is None:
                continue
            
            # 当セッション取得
            curr = get_session_data(df_1m, session_type, date)
            if curr is None:
                continue
            
            # エントリー価格
            entry = tick_round(curr['open'])
            prev_close = prev['close']
            
            # RiskGate (ギャップチェック)
            gap_rate = abs(entry - prev_close) / prev_close if prev_close > 0 else 0
            if gap_rate >= G_CUT:
                continue
            
            # 15分足データ (直前セッション)
            prev_15m = df_15m.loc[prev['data'].index[0]:prev['data'].index[-1]]
            
            # 直近10セッションのレンジ
            recent_ranges = pd.Series([r for r in session_ranges[-10:]] if len(session_ranges) >= 10 else [500])
            
            # Raptorシグナル判定
            signal = get_raptor_signal(
                prev_ohlc={'open': prev['open'], 'high': prev['high'], 'low': prev['low'], 'close': prev['close']},
                prev_15m_closes=prev_15m['Close'] if not prev_15m.empty else pd.Series(),
                recent_ranges=recent_ranges
            )
            
            # セッションレンジを履歴に追加
            session_ranges.append(prev['high'] - prev['low'])
            
            if signal['action'] is None:
                continue
            
            # トレード実行
            prev_range = prev['high'] - prev['low']
            trade = execute_trade(curr, signal['action'], entry, stop_mult, target_mult, prev_range)
            
            trades.append({
                'date': date,
                'session': session_type,
                'action': signal['action'],
                **trade,
                'score_b': signal['score_b'],
                'score_b_adj': signal['score_b_adj'],
                'score_c': signal['score_c'],
                'is_overheat': signal['is_overheat']
            })
    
    return trades


# ============================================================
# メイン
# ============================================================
def main():
    df_1m = load_data()
    
    # 15分足作成
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    
    print(f"\n🦖 Raptor225 バックテスト (内部プロンプト準拠)")
    print(f"   N={N_MOMENTUM}, r={R_OVERHEAT}, G_cut={G_CUT*100}%")
    print("\n" + "=" * 90)
    print(f"{'Stop':>5} | {'Tgt':>5} || {'Trades':>6} | {'Win%':>6} | {'総損益':>12} | {'月平均':>10} | {'TGT':>4} | {'STP':>4}")
    print("-" * 90)
    
    results = []
    
    for stop_mult, target_mult in product(STOP_RANGE, TARGET_RANGE):
        trades = run_backtest(df_1m, df_15m, stop_mult, target_mult)
        
        if not trades:
            continue
        
        df = pd.DataFrame(trades)
        total = len(df)
        wins = len(df[df['pnl'] > 0])
        total_pnl = df['pnl'].sum()
        monthly = total_pnl / 36  # 3年
        
        tgt_count = len(df[df['reason'] == 'TARGET'])
        stp_count = len(df[df['reason'] == 'STOP'])
        
        results.append({
            'stop': stop_mult,
            'target': target_mult,
            'trades': total,
            'win_rate': wins / total * 100,
            'total_pnl': total_pnl,
            'monthly': monthly,
            'details': df
        })
        
        print(f"{stop_mult:>5} | {target_mult:>5} || {total:>6} | {wins/total*100:>5.1f}% | "
              f"¥{total_pnl:>+10,.0f} | ¥{monthly:>+8,.0f} | {tgt_count:>4} | {stp_count:>4}")
    
    print("=" * 90)
    
    if results:
        best = max(results, key=lambda x: x['monthly'])
        print(f"\n🏆 ベスト: Stop={best['stop']} Target={best['target']}")
        print(f"   月平均: ¥{best['monthly']:+,.0f} | 総損益: ¥{best['total_pnl']:+,.0f}")
        
        # 2025年12月詳細
        print(f"\n{'='*100}")
        print(f"📅 2025年12月 詳細 (Stop={best['stop']} Target={best['target']})")
        print("=" * 100)
        
        dec = best['details'][best['details']['date'].apply(lambda d: d.year == 2025 and d.month == 12)]
        
        if len(dec) > 0:
            for _, t in dec.iterrows():
                overheat = "🔥" if t['is_overheat'] else ""
                print(f"  {t['date']} {t['session']:<5} | {t['action']:4} | B={t['score_b_adj']:+d} C={t['score_c']:+d}{overheat} | "
                      f"Entry:{t['entry']:,.0f}→Exit:{t['exit']:,.0f} | {t['reason']:<6} | PnL:¥{t['pnl']:+,.0f}")
        else:
            print("  (2025年12月のトレードなし)")


if __name__ == "__main__":
    main()
