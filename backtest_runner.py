#!/usr/bin/env python3
"""
Raptor225 Backtest
==================
楽天証券 / 資金10万円 / 日経225マイクロ1枚 / デイトレード

セッション構成:
  - DAY:   08:45〜15:15 (直前セッション = 前日NIGHT)
  - NIGHT: 16:30〜翌06:00 (直前セッション = 同日DAY)
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
    """バックテスト設定"""
    # 資金・ロット
    capital: int = 100_000
    lots: int = 1
    multiplier: int = 10  # 1ポイント = 10円
    commission: int = 22  # 往復手数料
    tick: int = 5         # 呼値
    
    # Raptorロジック
    gap_threshold: float = 0.0025  # 0.25%
    stop_atr: float = 1.0
    target_atr: float = 2.0
    
    # セッション時刻
    day_open: time = time(8, 45)
    day_close: time = time(15, 15)
    night_open: time = time(16, 30)
    night_close: time = time(6, 0)


CFG = Config()


# ============================================================
# ユーティリティ
# ============================================================
def tick_round(price: float) -> int:
    """5円刻みに丸める"""
    return int(round(price / CFG.tick) * CFG.tick)


def calc_slope(closes: pd.Series) -> float:
    """終値の回帰傾き"""
    n = len(closes)
    if n < 2:
        return 0.0
    x = np.arange(n)
    y = closes.values
    return float(np.polyfit(x, y, 1)[0])


# ============================================================
# データ読み込み
# ============================================================
def load_excel_data() -> pd.DataFrame:
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
    
    # カラム名を統一
    df.rename(columns={
        '日付': 'Date', '時間': 'Time', '時刻': 'Time',
        '始値': 'Open', '高値': 'High', '安値': 'Low', '終値': 'Close'
    }, inplace=True)
    
    # Datetime インデックス化
    df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df = df.set_index('Datetime').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df = df[['Open', 'High', 'Low', 'Close']].astype(float)
    
    print(f"✅ {len(df):,}本 ({df.index[0]} 〜 {df.index[-1]})")
    return df


def prepare_data(df_1m: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """15分足とATRを準備"""
    # 15分足
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    
    # 日足ATR (14日平均レンジ)
    df_daily = df_1m.resample('D').agg({'High': 'max', 'Low': 'min'}).dropna()
    df_daily['ATR'] = (df_daily['High'] - df_daily['Low']).rolling(14).mean()
    
    return df_15m, df_daily


# ============================================================
# セッション処理
# ============================================================
@dataclass
class Session:
    """セッション情報"""
    name: str
    open_time: datetime
    close_time: datetime
    prev_open: datetime
    prev_close: datetime


def get_day_session(date) -> Session:
    """DAYセッション情報を取得"""
    yesterday = date - timedelta(days=1)
    return Session(
        name='DAY',
        open_time=datetime.combine(date, CFG.day_open),
        close_time=datetime.combine(date, CFG.day_close),
        prev_open=datetime.combine(yesterday, CFG.night_open),
        prev_close=datetime.combine(date, CFG.night_close)
    )


def get_night_session(date) -> Session:
    """NIGHTセッション情報を取得"""
    tomorrow = date + timedelta(days=1)
    return Session(
        name='NIGHT',
        open_time=datetime.combine(date, CFG.night_open),
        close_time=datetime.combine(tomorrow, CFG.night_close),
        prev_open=datetime.combine(date, CFG.day_open),
        prev_close=datetime.combine(date, CFG.day_close)
    )


def get_session_ohlc(df: pd.DataFrame, start: datetime, end: datetime) -> Optional[Dict]:
    """セッションのOHLC取得"""
    data = df.loc[start:end]
    if data.empty or len(data) < 100:
        return None
    return {
        'open': data.iloc[0]['Open'],
        'high': data['High'].max(),
        'low': data['Low'].min(),
        'close': data.iloc[-1]['Close'],
        'data': data
    }


# ============================================================
# Raptorロジック
# ============================================================
def get_raptor_signal(prev_ohlc: Dict, slope: float) -> Optional[str]:
    """
    Raptorシグナル判定
    
    B: 直前セッションの方向 (陽線+1, 陰線-1)
    C: モメンタム傾き (正+1, 負-1)
    
    B + C >= +2 → BUY
    B + C <= -2 → SELL
    """
    # B: 直前セッションの方向
    if prev_ohlc['close'] > prev_ohlc['open']:
        score_b = 1
    elif prev_ohlc['close'] < prev_ohlc['open']:
        score_b = -1
    else:
        score_b = 0
    
    # C: モメンタム
    score_c = 1 if slope > 0 else -1
    
    total = score_b + score_c
    
    if total >= 2:
        return 'BUY'
    elif total <= -2:
        return 'SELL'
    return None


def execute_trade(
    session_data: pd.DataFrame,
    action: str,
    entry: int,
    stop: int,
    target: int
) -> Tuple[int, str]:
    """
    トレード実行
    
    Returns:
        (exit_price, reason)
        reason: 'TARGET', 'STOP', 'CLOSE'
    """
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
    return tick_round(session_data.iloc[-1]['Close']), 'CLOSE'


# ============================================================
# バックテスト本体
# ============================================================
def process_session(
    session: Session,
    df_1m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_daily: pd.DataFrame
) -> Optional[Dict]:
    """1セッションを処理"""
    
    # 直前セッションデータ取得
    prev = get_session_ohlc(df_1m, session.prev_open, session.prev_close)
    if prev is None:
        return None
    
    # 当セッションデータ取得
    curr = get_session_ohlc(df_1m, session.open_time, session.close_time)
    if curr is None:
        return None
    
    # エントリー価格
    entry = tick_round(curr['data'].iloc[0]['Open'])
    
    # ギャップチェック
    gap = abs(entry - prev['close']) / prev['close']
    if gap >= CFG.gap_threshold:
        return None
    
    # モメンタム計算
    prev_15m = df_15m.loc[session.prev_open:session.prev_close]
    if len(prev_15m) < 10:
        return None
    
    slope = calc_slope(prev_15m['Close'])
    
    # シグナル判定
    action = get_raptor_signal(prev, slope)
    if action is None:
        return None
    
    # ATR取得
    atr_date = session.prev_close.date()
    try:
        atr = df_daily.loc[:str(atr_date)]['ATR'].iloc[-1]
        if pd.isna(atr) or atr <= 0:
            atr = 400
    except:
        atr = 400
    
    # Stop/Target計算
    s_dist = tick_round(atr * CFG.stop_atr)
    t_dist = tick_round(atr * CFG.target_atr)
    
    if action == 'BUY':
        stop = entry - s_dist
        target = entry + t_dist
    else:
        stop = entry + s_dist
        target = entry - t_dist
    
    # トレード実行
    exit_price, reason = execute_trade(curr['data'], action, entry, stop, target)
    
    # 損益計算
    diff = (exit_price - entry) if action == 'BUY' else (entry - exit_price)
    pnl = diff * CFG.multiplier * CFG.lots - CFG.commission
    
    return {
        'date': session.open_time.date(),
        'session': session.name,
        'action': action,
        'entry': entry,
        'exit': exit_price,
        'pnl': pnl,
        'reason': reason
    }


def run_backtest(df_1m: pd.DataFrame) -> List[Dict]:
    """バックテスト実行"""
    df_15m, df_daily = prepare_data(df_1m)
    dates = sorted(set(df_1m.index.date))
    
    trades = []
    
    print(f"\n🚀 バックテスト開始 ({len(dates)}日)")
    print(f"   マイクロ{CFG.lots}枚, Stop {CFG.stop_atr} ATR, Target {CFG.target_atr} ATR")
    print("-" * 70)
    
    for date in dates:
        # DAYセッション
        day = get_day_session(date)
        result = process_session(day, df_1m, df_15m, df_daily)
        if result:
            trades.append(result)
            if len(trades) <= 5:
                print(f"  #{len(trades)} {result['date']} DAY {result['action']} "
                      f"{result['entry']}→{result['exit']} PnL={result['pnl']:+,.0f}")
        
        # NIGHTセッション
        night = get_night_session(date)
        result = process_session(night, df_1m, df_15m, df_daily)
        if result:
            trades.append(result)
            if len(trades) <= 5:
                print(f"  #{len(trades)} {result['date']} NIGHT {result['action']} "
                      f"{result['entry']}→{result['exit']} PnL={result['pnl']:+,.0f}")
    
    return trades


def print_results(trades: List[Dict]):
    """結果出力"""
    print("-" * 70)
    
    if not trades:
        print("⚠️ トレードなし")
        return
    
    df = pd.DataFrame(trades)
    
    # 統計
    total = len(df)
    wins = len(df[df['pnl'] > 0])
    win_rate = wins / total * 100
    
    gross_win = df[df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df[df['pnl'] <= 0]['pnl'].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    
    total_pnl = df['pnl'].sum()
    final_capital = CFG.capital + total_pnl
    monthly = total_pnl / 96  # 約8年
    
    day_trades = len(df[df['session'] == 'DAY'])
    night_trades = len(df[df['session'] == 'NIGHT'])
    
    print(f"\n📊 結果")
    print("=" * 70)
    print(f"  期間        : {df['date'].min()} 〜 {df['date'].max()}")
    print(f"  トレード数  : {total}回 (DAY:{day_trades} NIGHT:{night_trades})")
    print(f"  勝率        : {win_rate:.1f}% ({wins}勝 {total-wins}敗)")
    print(f"  PF          : {pf:.2f}")
    print("=" * 70)
    print(f"  最終資金    : ¥{final_capital:,.0f}")
    print(f"  純損益      : ¥{total_pnl:+,.0f}")
    print(f"  リターン    : {(final_capital - CFG.capital) / CFG.capital * 100:+.1f}%")
    print(f"  月平均      : ¥{monthly:+,.0f}")
    print("=" * 70)
    
    # 決済理由
    reason_counts = df['reason'].value_counts()
    print(f"\n📈 決済理由:")
    for reason, count in reason_counts.items():
        print(f"  {reason}: {count}回 ({count/total*100:.1f}%)")


# ============================================================
# メイン
# ============================================================
def main():
    df_1m = load_excel_data()
    trades = run_backtest(df_1m)
    print_results(trades)


if __name__ == "__main__":
    main()
