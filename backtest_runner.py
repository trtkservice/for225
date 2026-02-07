"""
Raptor225 Backtest Runner (Clean Rewrite v2)
=============================================
運用環境: 楽天証券 / 資金10万円 / 日経225マイクロ 1枚 固定
対象: DAYセッション + NIGHTセッション (両方エントリー)
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
INITIAL_CAPITAL = 100_000   # 資金10万円
LOTS = 1                    # マイクロ 1枚
MULTIPLIER = 10             # マイクロ = 1ポイント10円
COST_PER_TRADE = 22         # 楽天マイクロ往復手数料 (11円 x 2)
TICK_SIZE = 5               # 呼値

# Raptorロジック設定
GAP_THRESHOLD = 0.0025      # 0.25%
MOMENTUM_PERIOD = 48        # 12時間分 (15分足48本)
STOP_ATR_MULT = 1.0         # ストップ = 1.0 ATR
TARGET_ATR_MULT = 2.0       # ターゲット = 2.0 ATR

# ============================================================
# ユーティリティ
# ============================================================
def round_to_tick(price):
    """価格を呼値(5円刻み)に丸める"""
    return int(round(price / TICK_SIZE) * TICK_SIZE)

def calculate_slope(series):
    """終値系列の回帰直線の傾きを計算"""
    y = series.values
    x = np.arange(len(y))
    if len(x) < 2:
        return 0
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope

# ============================================================
# データ読み込み
# ============================================================
def load_data():
    """N225minif_*.xlsx を全て読み込み、1分足DataFrameを返す"""
    data_dir = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(data_dir, "N225minif_*.xlsx")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("❌ エラー: N225minif_*.xlsx ファイルが見つかりません")
        sys.exit(1)
    
    print(f"📥 {len(files)}個のExcelファイルを読み込み中...")
    
    dfs = []
    for f in files:
        print(f"   {os.path.basename(f)}")
        try:
            df = pd.read_excel(f)
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ 読み込み失敗: {e}")
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    # カラム名を統一
    rename_map = {
        '日付': 'Date', '時間': 'Time', '時刻': 'Time',
        '始値': 'Open', '高値': 'High', '安値': 'Low', '終値': 'Close'
    }
    full_df.rename(columns=rename_map, inplace=True)
    
    # Datetime インデックス作成
    full_df['Datetime'] = pd.to_datetime(
        full_df['Date'].astype(str) + ' ' + full_df['Time'].astype(str)
    )
    full_df.set_index('Datetime', inplace=True)
    full_df.sort_index(inplace=True)
    full_df = full_df[~full_df.index.duplicated(keep='first')]
    
    df = full_df[['Open', 'High', 'Low', 'Close']].astype(float)
    print(f"✅ {len(df):,}本の1分足を読み込み完了 ({df.index[0]} 〜 {df.index[-1]})")
    
    return df

# ============================================================
# セッション単位のトレード実行
# ============================================================
def execute_session_trade(df_1m, df_15m, df_daily, session_type, 
                          session_open_time, session_close_time,
                          prev_session_close_time):
    """
    1セッション分のトレードをシミュレート
    
    Returns:
        dict: トレード結果 (None if no trade)
    """
    
    # セッションデータ取得
    session_data = df_1m.loc[session_open_time:session_close_time]
    if session_data.empty:
        return None
    
    entry_price = round_to_tick(session_data.iloc[0]['Open'])
    
    # 直前セッションの終値を取得
    # prev_session_close_time より前のデータで最後の終値
    prev_data = df_1m.loc[:prev_session_close_time]
    if prev_data.empty:
        return None
    
    prev_close = prev_data.iloc[-1]['Close']
    
    # 直前セッションの始値を取得 (陽線/陰線判定用)
    # DAYの場合: 直前はNIGHT (前日16:30〜当日06:00)
    # NIGHTの場合: 直前はDAY (当日08:45〜15:15)
    
    if session_type == "DAY":
        # 直前NIGHT: 前日16:30開始
        prev_open_time = prev_session_close_time - timedelta(hours=13, minutes=30)
    else:  # NIGHT
        # 直前DAY: 同日08:45開始
        prev_open_time = prev_session_close_time - timedelta(hours=6, minutes=30)
    
    prev_session_data = df_1m.loc[prev_open_time:prev_session_close_time]
    if prev_session_data.empty:
        return None
    
    prev_open = prev_session_data.iloc[0]['Open']
    
    # ===== Raptorロジック =====
    
    # 1. ギャップチェック (RiskGate)
    gap_rate = abs(entry_price - prev_close) / prev_close
    if gap_rate >= GAP_THRESHOLD:
        return None  # ギャップが大きすぎる → 見送り
    
    # 2. B判定: 直前セッションのトレンド (陽線/陰線)
    if prev_close > prev_open:
        score_b = 1   # 陽線
    elif prev_close < prev_open:
        score_b = -1  # 陰線
    else:
        score_b = 0   # 同値
    
    # 3. C判定: モメンタム (15分足の傾き)
    # 直前セッション終了時点までの15分足を使う
    recent_15m = df_15m.loc[:prev_session_close_time].iloc[-MOMENTUM_PERIOD:]
    if len(recent_15m) < MOMENTUM_PERIOD // 2:
        return None  # データ不足
    
    slope = calculate_slope(recent_15m['Close'])
    score_c = 1 if slope > 0 else -1
    
    # 4. 合計スコア → シグナル決定
    total_score = score_b + score_c
    
    if total_score >= 2:
        action = "BUY"
    elif total_score <= -2:
        action = "SELL"
    else:
        return None  # NO-TRADE
    
    # ===== リスク管理 =====
    
    # ATR取得 (直前日ベース)
    prev_date = prev_session_close_time.date()
    try:
        ts = pd.Timestamp(prev_date)
        if not isinstance(df_daily.index, pd.DatetimeIndex):
            df_daily.index = pd.to_datetime(df_daily.index)
        idx = df_daily.index.get_indexer([ts], method='pad')[0]
        atr = df_daily.iloc[idx]['ATR'] if idx != -1 else 500
        if pd.isna(atr) or atr <= 0:
            atr = 500
    except:
        atr = 500
    
    # ストップ・ターゲット計算
    stop_dist = round_to_tick(atr * STOP_ATR_MULT)
    target_dist = round_to_tick(atr * TARGET_ATR_MULT)
    
    if action == "BUY":
        stop = entry_price - stop_dist
        target = entry_price + target_dist
    else:
        stop = entry_price + stop_dist
        target = entry_price - target_dist
    
    # ===== トレード実行 (分足シミュレーション) =====
    exit_price = None
    
    for _, row in session_data.iterrows():
        if action == "BUY":
            if row['Low'] <= stop:
                exit_price = stop
                break
            if row['High'] >= target:
                exit_price = target
                break
        else:  # SELL
            if row['High'] >= stop:
                exit_price = stop
                break
            if row['Low'] <= target:
                exit_price = target
                break
    
    # ヒットしなければ引けで決済
    if exit_price is None:
        exit_price = round_to_tick(session_data.iloc[-1]['Close'])
    
    # ===== 損益計算 =====
    if action == "BUY":
        diff = exit_price - entry_price
    else:
        diff = entry_price - exit_price
    
    # 損益 = 値幅 × 倍率 × 枚数 - 手数料
    pnl = (diff * MULTIPLIER * LOTS) - COST_PER_TRADE
    
    return {
        'date': session_open_time.date(),
        'session': session_type,
        'action': action,
        'entry': entry_price,
        'exit': exit_price,
        'diff': diff,
        'pnl': pnl
    }

# ============================================================
# バックテスト本体
# ============================================================
def run_backtest(df_1m):
    """Raptorロジックでバックテストを実行 (DAY + NIGHT)"""
    
    # 15分足を作成
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    
    # 日次ATRを作成 (14日移動平均)
    df_daily = df_1m.resample('D').agg({
        'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_daily['ATR'] = (df_daily['High'] - df_daily['Low']).rolling(14).mean()
    
    # 取引日リスト
    unique_dates = sorted(set(df_1m.index.date))
    
    # 結果格納
    capital = INITIAL_CAPITAL
    trades = []
    
    print(f"\n🚀 バックテスト開始 (対象: {len(unique_dates)}日 × DAY/NIGHT)")
    print(f"   設定: マイクロ{LOTS}枚, Stop {STOP_ATR_MULT} ATR, Target {TARGET_ATR_MULT} ATR")
    print("-" * 60)
    
    for i in range(1, len(unique_dates)):
        curr_date = unique_dates[i]
        prev_date = unique_dates[i - 1]
        
        # ===== DAYセッション =====
        # エントリー: 08:45, 決済: 15:15
        # 直前セッション: 前日NIGHT (06:00終了)
        day_open = datetime.combine(curr_date, time(8, 45))
        day_close = datetime.combine(curr_date, time(15, 15))
        prev_night_close = datetime.combine(curr_date, time(6, 0))
        
        result = execute_session_trade(
            df_1m, df_15m, df_daily, 
            "DAY", day_open, day_close, prev_night_close
        )
        if result:
            capital += result['pnl']
            trades.append(result)
            if len(trades) <= 5:
                print(f"  #{len(trades)} {result['date']} {result['session']} {result['action']} Entry={result['entry']} Exit={result['exit']} PnL={result['pnl']:+,.0f}円")
        
        # ===== NIGHTセッション =====
        # エントリー: 16:30, 決済: 翌日06:00
        # 直前セッション: 同日DAY (15:15終了)
        night_open = datetime.combine(curr_date, time(16, 30))
        night_close = datetime.combine(curr_date + timedelta(days=1), time(6, 0))
        prev_day_close = datetime.combine(curr_date, time(15, 15))
        
        result = execute_session_trade(
            df_1m, df_15m, df_daily,
            "NIGHT", night_open, night_close, prev_day_close
        )
        if result:
            capital += result['pnl']
            trades.append(result)
            if len(trades) <= 5:
                print(f"  #{len(trades)} {result['date']} {result['session']} {result['action']} Entry={result['entry']} Exit={result['exit']} PnL={result['pnl']:+,.0f}円")
    
    # ===== 結果集計 =====
    print("-" * 60)
    
    if not trades:
        print("⚠️ トレードが発生しませんでした")
        return
    
    df_trades = pd.DataFrame(trades)
    
    total_pnl = df_trades['pnl'].sum()
    win_trades = df_trades[df_trades['pnl'] > 0]
    lose_trades = df_trades[df_trades['pnl'] <= 0]
    
    win_count = len(win_trades)
    lose_count = len(lose_trades)
    total_count = len(df_trades)
    win_rate = win_count / total_count * 100
    
    gross_profit = win_trades['pnl'].sum() if len(win_trades) > 0 else 0
    gross_loss = abs(lose_trades['pnl'].sum()) if len(lose_trades) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # 月数計算
    months = len(set([t['date'].strftime('%Y-%m') for t in trades]))
    avg_monthly_pnl = total_pnl / months if months > 0 else 0
    
    return_pct = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    # セッション別集計
    day_trades = df_trades[df_trades['session'] == 'DAY']
    night_trades = df_trades[df_trades['session'] == 'NIGHT']
    
    print(f"\n📊 バックテスト結果")
    print(f"{'='*60}")
    print(f"  期間          : {df_trades['date'].min()} 〜 {df_trades['date'].max()}")
    print(f"  トレード数    : {total_count}回 (DAY:{len(day_trades)} NIGHT:{len(night_trades)})")
    print(f"  勝率          : {win_rate:.1f}% ({win_count}勝 {lose_count}敗)")
    print(f"  プロフィット  : {profit_factor:.2f}")
    print(f"{'='*60}")
    print(f"  最終資金      : ¥{capital:,.0f}")
    print(f"  純損益        : ¥{total_pnl:+,.0f}")
    print(f"  リターン      : {return_pct:+.1f}%")
    print(f"  月平均損益    : ¥{avg_monthly_pnl:+,.0f}")
    print(f"{'='*60}")

# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    df = load_data()
    run_backtest(df)
