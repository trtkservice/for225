#!/usr/bin/env python3
"""
Raptor225 Backtest - NIGHT診断版
=================================
NIGHTセッションがなぜ負けるのかを徹底調査
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from datetime import datetime, time, timedelta
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# 設定
# ============================================================
@dataclass
class Config:
    capital: int = 100_000
    lots: int = 1
    multiplier: int = 10
    commission: int = 22
    tick: int = 5
    gap_threshold: float = 0.0025
    stop_atr: float = 1.0
    target_atr: float = 2.0
    day_open: time = time(8, 45)
    day_close: time = time(15, 15)
    night_open: time = time(16, 30)
    night_close: time = time(6, 0)


CFG = Config()


def tick_round(price: float) -> int:
    return int(round(price / CFG.tick) * CFG.tick)


def calc_slope(closes: pd.Series) -> float:
    n = len(closes)
    if n < 2:
        return 0.0
    x = np.arange(n)
    y = closes.values
    return float(np.polyfit(x, y, 1)[0])


def load_excel_data() -> pd.DataFrame:
    base = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(base, "N225minif_*.xlsx")))
    
    if not files:
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
    
    print(f"✅ {len(df):,}本")
    
    # 直近3年に絞る
    df = df[df.index >= '2023-01-01']
    print(f"📅 2023年以降: {len(df):,}本")
    
    return df


def prepare_data(df_1m):
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    
    df_daily = df_1m.resample('D').agg({'High': 'max', 'Low': 'min'}).dropna()
    df_daily['ATR'] = (df_daily['High'] - df_daily['Low']).rolling(14).mean()
    
    return df_15m, df_daily


def get_signal(prev_close, prev_open, slope):
    score_b = 1 if prev_close > prev_open else -1 if prev_close < prev_open else 0
    score_c = 1 if slope > 0 else -1
    total = score_b + score_c
    
    if total >= 2:
        return 'BUY', score_b, score_c
    elif total <= -2:
        return 'SELL', score_b, score_c
    return None, score_b, score_c


def execute_trade(session_data, action, entry, stop, target):
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


def run_diagnosis(df_1m):
    df_15m, df_daily = prepare_data(df_1m)
    dates = sorted(set(df_1m.index.date))
    
    day_trades = []
    night_trades = []
    
    # 診断用サンプル (2025年12月のNIGHT)
    diag_samples = []
    
    for date in dates:
        # ========== DAYセッション ==========
        yesterday = date - timedelta(days=1)
        prev_start = datetime.combine(yesterday, CFG.night_open)
        prev_end = datetime.combine(date, CFG.night_close)
        session_start = datetime.combine(date, CFG.day_open)
        session_end = datetime.combine(date, CFG.day_close)
        
        prev_data = df_1m.loc[prev_start:prev_end]
        session_data = df_1m.loc[session_start:session_end]
        
        if len(prev_data) >= 100 and len(session_data) >= 100:
            prev_open = prev_data.iloc[0]['Open']
            prev_close = prev_data.iloc[-1]['Close']
            entry = tick_round(session_data.iloc[0]['Open'])
            
            gap = abs(entry - prev_close) / prev_close
            if gap < CFG.gap_threshold:
                prev_15m = df_15m.loc[prev_start:prev_end]
                if len(prev_15m) >= 10:
                    slope = calc_slope(prev_15m['Close'])
                    action, b, c = get_signal(prev_close, prev_open, slope)
                    
                    if action:
                        try:
                            atr = df_daily.loc[:str(yesterday)]['ATR'].iloc[-1]
                            if pd.isna(atr) or atr <= 0:
                                atr = 400
                        except:
                            atr = 400
                        
                        s_dist = tick_round(atr * CFG.stop_atr)
                        t_dist = tick_round(atr * CFG.target_atr)
                        
                        if action == 'BUY':
                            stop, target = entry - s_dist, entry + t_dist
                        else:
                            stop, target = entry + s_dist, entry - t_dist
                        
                        exit_price, reason = execute_trade(session_data, action, entry, stop, target)
                        diff = (exit_price - entry) if action == 'BUY' else (entry - exit_price)
                        pnl = diff * CFG.multiplier * CFG.lots - CFG.commission
                        
                        # 実際の値動き
                        actual_move = session_data.iloc[-1]['Close'] - session_data.iloc[0]['Open']
                        
                        day_trades.append({
                            'date': date, 'action': action, 'pnl': pnl, 'reason': reason,
                            'b': b, 'c': c, 'slope': slope,
                            'prev_move': prev_close - prev_open,
                            'actual_move': actual_move,
                            'signal_correct': (action == 'BUY' and actual_move > 0) or (action == 'SELL' and actual_move < 0)
                        })
        
        # ========== NIGHTセッション ==========
        tomorrow = date + timedelta(days=1)
        prev_start = datetime.combine(date, CFG.day_open)
        prev_end = datetime.combine(date, CFG.day_close)
        session_start = datetime.combine(date, CFG.night_open)
        session_end = datetime.combine(tomorrow, CFG.night_close)
        
        prev_data = df_1m.loc[prev_start:prev_end]
        session_data = df_1m.loc[session_start:session_end]
        
        if len(prev_data) >= 100 and len(session_data) >= 100:
            prev_open = prev_data.iloc[0]['Open']
            prev_close = prev_data.iloc[-1]['Close']
            entry = tick_round(session_data.iloc[0]['Open'])
            
            gap = abs(entry - prev_close) / prev_close
            if gap < CFG.gap_threshold:
                prev_15m = df_15m.loc[prev_start:prev_end]
                if len(prev_15m) >= 10:
                    slope = calc_slope(prev_15m['Close'])
                    action, b, c = get_signal(prev_close, prev_open, slope)
                    
                    if action:
                        try:
                            atr = df_daily.loc[:str(date)]['ATR'].iloc[-1]
                            if pd.isna(atr) or atr <= 0:
                                atr = 400
                        except:
                            atr = 400
                        
                        s_dist = tick_round(atr * CFG.stop_atr)
                        t_dist = tick_round(atr * CFG.target_atr)
                        
                        if action == 'BUY':
                            stop, target = entry - s_dist, entry + t_dist
                        else:
                            stop, target = entry + s_dist, entry - t_dist
                        
                        exit_price, reason = execute_trade(session_data, action, entry, stop, target)
                        diff = (exit_price - entry) if action == 'BUY' else (entry - exit_price)
                        pnl = diff * CFG.multiplier * CFG.lots - CFG.commission
                        
                        actual_move = session_data.iloc[-1]['Close'] - session_data.iloc[0]['Open']
                        
                        night_trades.append({
                            'date': date, 'action': action, 'pnl': pnl, 'reason': reason,
                            'b': b, 'c': c, 'slope': slope,
                            'prev_move': prev_close - prev_open,
                            'actual_move': actual_move,
                            'signal_correct': (action == 'BUY' and actual_move > 0) or (action == 'SELL' and actual_move < 0)
                        })
                        
                        # 2025年12月のサンプル
                        if date.year == 2025 and date.month == 12:
                            diag_samples.append({
                                'date': date,
                                'prev_open': prev_open,
                                'prev_close': prev_close,
                                'prev_move': prev_close - prev_open,
                                'slope': slope,
                                'b': b, 'c': c,
                                'action': action,
                                'entry': entry,
                                'exit': exit_price,
                                'actual_move': actual_move,
                                'pnl': pnl,
                                'signal_correct': (action == 'BUY' and actual_move > 0) or (action == 'SELL' and actual_move < 0)
                            })
    
    # ========== 分析出力 ==========
    print("\n" + "=" * 80)
    print("📊 DAY vs NIGHT 詳細比較")
    print("=" * 80)
    
    df_day = pd.DataFrame(day_trades) if day_trades else pd.DataFrame()
    df_night = pd.DataFrame(night_trades) if night_trades else pd.DataFrame()
    
    print(f"\n【基本統計】")
    print(f"{'':15} {'DAY':>15} {'NIGHT':>15}")
    print("-" * 50)
    print(f"{'トレード数':15} {len(df_day):>15} {len(df_night):>15}")
    
    if len(df_day) > 0:
        day_wins = len(df_day[df_day['pnl'] > 0])
        day_winrate = day_wins / len(df_day) * 100
        day_pnl = df_day['pnl'].sum()
        day_avg = df_day['pnl'].mean()
        day_correct = df_day['signal_correct'].sum() / len(df_day) * 100
    else:
        day_winrate = day_pnl = day_avg = day_correct = 0
    
    if len(df_night) > 0:
        night_wins = len(df_night[df_night['pnl'] > 0])
        night_winrate = night_wins / len(df_night) * 100
        night_pnl = df_night['pnl'].sum()
        night_avg = df_night['pnl'].mean()
        night_correct = df_night['signal_correct'].sum() / len(df_night) * 100
    else:
        night_winrate = night_pnl = night_avg = night_correct = 0
    
    print(f"{'勝率':15} {day_winrate:>14.1f}% {night_winrate:>14.1f}%")
    print(f"{'シグナル正解率':15} {day_correct:>14.1f}% {night_correct:>14.1f}%")
    print(f"{'総損益':15} {'¥'+f'{day_pnl:+,.0f}':>14} {'¥'+f'{night_pnl:+,.0f}':>14}")
    print(f"{'平均損益':15} {'¥'+f'{day_avg:+,.0f}':>14} {'¥'+f'{night_avg:+,.0f}':>14}")
    
    # シグナル方向ごとの分析
    print(f"\n【シグナル方向別】")
    for session_name, df in [('DAY', df_day), ('NIGHT', df_night)]:
        if len(df) == 0:
            continue
        print(f"\n{session_name}:")
        for action in ['BUY', 'SELL']:
            sub = df[df['action'] == action]
            if len(sub) > 0:
                wins = len(sub[sub['pnl'] > 0])
                correct = sub['signal_correct'].sum()
                avg_pnl = sub['pnl'].mean()
                print(f"  {action}: {len(sub)}回, 勝率{wins/len(sub)*100:.1f}%, 正解率{correct/len(sub)*100:.1f}%, 平均¥{avg_pnl:+,.0f}")
    
    # 2025年12月のNIGHTサンプル
    print(f"\n【2025年12月 NIGHTサンプル】")
    print("-" * 100)
    for s in diag_samples:
        direction_match = "✅" if s['signal_correct'] else "❌"
        print(f"{s['date']} | DAY動き:{s['prev_move']:+.0f} → B={s['b']:+d} | "
              f"Slope:{s['slope']:+.2f} → C={s['c']:+d} | "
              f"Signal:{s['action']} | NIGHT動き:{s['actual_move']:+.0f} {direction_match} | "
              f"PnL:¥{s['pnl']:+,.0f}")
    
    # 核心的な問題
    print(f"\n{'='*80}")
    print("🔍 核心的な問題の特定")
    print("=" * 80)
    
    if len(df_night) > 0:
        # NIGHTでBUYした時、実際に上がったか？
        night_buy = df_night[df_night['action'] == 'BUY']
        night_sell = df_night[df_night['action'] == 'SELL']
        
        if len(night_buy) > 0:
            buy_up = len(night_buy[night_buy['actual_move'] > 0])
            print(f"  NIGHT BUY時に実際に上昇: {buy_up}/{len(night_buy)} ({buy_up/len(night_buy)*100:.1f}%)")
        
        if len(night_sell) > 0:
            sell_down = len(night_sell[night_sell['actual_move'] < 0])
            print(f"  NIGHT SELL時に実際に下落: {sell_down}/{len(night_sell)} ({sell_down/len(night_sell)*100:.1f}%)")
        
        # DAYのトレンドがNIGHTに続くか？
        print(f"\n  DAYトレンド → NIGHT継続率:")
        day_up_night_up = len(df_night[(df_night['prev_move'] > 0) & (df_night['actual_move'] > 0)])
        day_up = len(df_night[df_night['prev_move'] > 0])
        day_down_night_down = len(df_night[(df_night['prev_move'] < 0) & (df_night['actual_move'] < 0)])
        day_down = len(df_night[df_night['prev_move'] < 0])
        
        if day_up > 0:
            print(f"    DAY↑ → NIGHT↑: {day_up_night_up}/{day_up} ({day_up_night_up/day_up*100:.1f}%)")
        if day_down > 0:
            print(f"    DAY↓ → NIGHT↓: {day_down_night_down}/{day_down} ({day_down_night_down/day_down*100:.1f}%)")


if __name__ == "__main__":
    df = load_excel_data()
    run_diagnosis(df)
