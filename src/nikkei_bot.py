#!/usr/bin/env python3
"""
Raptor225 Trading Bot
=====================
Raptor内部プロンプトと完全一致する判定ロジック

判定ロジック:
- B: 直前セッションの実体方向 (陽線+1, 陰線-1, 同値0)
- C: 直近15M 32本の回帰傾き (正+1, 負-1, 微小0)
- D: 過熱判定 (直前レンジ >= 直近10セッション平均×1.8 なら B→0へ弱める)
- TotalScore = B + C
- TotalScore >= +2 → BUY
- TotalScore <= -2 → SELL
- それ以外 → NO-TRADE

ギャップ:
- gap_rate = |entry - prev_close| / prev_close
- |gap_rate| >= 0.25% → NO-TRADE (RiskGate)

セッション時刻:
- DAY: 08:45開始, 15:45終了
- NIGHT: 16:30開始, 06:00終了
"""

import os
import sys
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import pytz
from datetime import datetime, time, timedelta
from pathlib import Path

# ============================================================
# 設定
# ============================================================
class Config:
    # タイムゾーン
    JST = pytz.timezone('Asia/Tokyo')
    
    # 運用設定
    CAPITAL = 100_000      # 資金10万円
    LOTS = 1               # マイクロ1枚
    MULTIPLIER = 10        # 1ポイント = 10円
    COMMISSION = 0         # Raptor準拠: cost=0
    TICK = 5               # 呼値
    
    # Raptorパラメータ
    G_CUT = 0.0025         # ギャップ閾値 0.25%
    N_MOMENTUM = 32        # モメンタム計算に使う15分足の本数
    R_OVERHEAT = 1.8       # 過熱判定倍率
    SLOPE_THRESHOLD = 0.5  # 傾きがこれ以下なら「微小」→ C=0
    
    # セッション時刻
    DAY_OPEN = time(8, 45)
    DAY_CLOSE = time(15, 45)   # Raptor準拠: 15:45
    NIGHT_OPEN = time(16, 30)
    NIGHT_CLOSE = time(6, 0)
    
    # データファイル
    DATA_FILE = "data/portfolio.json"
    TICKER = "NKD=F"


# ============================================================
# ユーティリティ
# ============================================================
def tick_round(price: float) -> int:
    """5円刻みに丸める"""
    return int(round(price / Config.TICK) * Config.TICK)


def calc_slope(closes: pd.Series) -> float:
    """終値配列の回帰傾き"""
    n = len(closes)
    if n < 2:
        return 0.0
    x = np.arange(n)
    y = closes.values
    return float(np.polyfit(x, y, 1)[0])


def send_line(message: str):
    """LINE Notify送信"""
    token = os.environ.get("LINE_NOTIFY_TOKEN")
    if not token:
        print(f"📱 (LINE未設定) {message}")
        return
    
    try:
        requests.post(
            "https://notify-api.line.me/api/notify",
            headers={"Authorization": f"Bearer {token}"},
            data={"message": f"\n{message}"},
            timeout=10
        )
    except Exception as e:
        print(f"⚠️ LINE送信失敗: {e}")


# ============================================================
# データ取得
# ============================================================
class MarketData:
    @staticmethod
    def fetch():
        """Yahoo Financeから日経225先物データを取得"""
        try:
            ticker = yf.Ticker(Config.TICKER)
            
            # 日足 (セッションOHLC用)
            daily = ticker.history(period="1mo")
            
            # 15分足 (モメンタム計算用)
            intraday = ticker.history(period="5d", interval="15m")
            
            return {
                'daily': daily,
                'intraday': intraday,
                'current_price': daily.iloc[-1]['Close'] if not daily.empty else None
            }
        except Exception as e:
            print(f"❌ データ取得失敗: {e}")
            return None


# ============================================================
# Raptorロジック
# ============================================================
class RaptorEngine:
    """Raptor225の判定ロジック (内部プロンプト完全準拠)"""
    
    def __init__(self, market_data: dict):
        self.daily = market_data.get('daily')
        self.intraday = market_data.get('intraday')
        self.current_price = market_data.get('current_price')
    
    def get_prev_session_ohlc(self, session: str) -> dict:
        """
        直前セッションのOHLC取得
        - DAY: 前日のNIGHT (最後から2番目の日足を使用)
        - NIGHT: 同日のDAY (最後の日足を使用)
        """
        if self.daily is None or len(self.daily) < 2:
            return None
        
        # 簡易的に日足で代用
        if session == 'DAY':
            prev = self.daily.iloc[-2]  # 前日
        else:
            prev = self.daily.iloc[-1]  # 当日
        
        return {
            'open': prev['Open'],
            'high': prev['High'],
            'low': prev['Low'],
            'close': prev['Close'],
            'range': prev['High'] - prev['Low']
        }
    
    def get_avg_range(self, n: int = 10) -> float:
        """直近nセッションの平均レンジ"""
        if self.daily is None or len(self.daily) < n:
            return 500  # デフォルト
        
        ranges = self.daily['High'].iloc[-n:] - self.daily['Low'].iloc[-n:]
        return ranges.mean()
    
    def calc_score_b(self, prev_ohlc: dict) -> int:
        """
        B: 直前セッションの実体方向
        - 陽線: +1
        - 陰線: -1
        - 同値: 0
        """
        if prev_ohlc['close'] > prev_ohlc['open']:
            return 1
        elif prev_ohlc['close'] < prev_ohlc['open']:
            return -1
        return 0
    
    def calc_score_c(self) -> int:
        """
        C: 直近15M 32本の回帰傾き
        - 正: +1
        - 負: -1
        - 微小: 0
        """
        if self.intraday is None or len(self.intraday) < Config.N_MOMENTUM:
            return 0
        
        closes = self.intraday['Close'].iloc[-Config.N_MOMENTUM:]
        slope = calc_slope(closes)
        
        # 微小判定
        if abs(slope) < Config.SLOPE_THRESHOLD:
            return 0
        
        return 1 if slope > 0 else -1
    
    def apply_overheat_d(self, score_b: int, prev_range: float, avg_range: float) -> int:
        """
        D: 過熱判定
        直前セッションレンジ >= 平均レンジ × r倍 なら B → 0 へ弱める
        """
        if avg_range <= 0:
            return score_b
        
        is_overheat = prev_range >= avg_range * Config.R_OVERHEAT
        
        if is_overheat:
            # Bを0に寄せる (1→0, -1→0, 0→0)
            return 0
        
        return score_b
    
    def check_risk_gate(self, entry_price: float, prev_close: float) -> tuple:
        """
        RiskGate: ギャップが大きすぎる場合は NO-TRADE
        Returns: (pass: bool, gap_rate: float)
        """
        if prev_close <= 0:
            return False, 0
        
        gap_rate = abs(entry_price - prev_close) / prev_close
        return gap_rate < Config.G_CUT, gap_rate
    
    def get_signal(self, session: str, entry_price: float) -> dict:
        """
        Raptorシグナル判定
        
        Args:
            session: 'DAY' or 'NIGHT'
            entry_price: 寄付き予想価格
        
        Returns:
            dict with signal details
        """
        result = {
            'session': session,
            'entry_price': entry_price,
            'verdict': 'NO-TRADE',
            'reason': '',
            'score_b': 0,
            'score_c': 0,
            'score_b_adj': 0,  # 過熱調整後
            'total_score': 0,
            'gap_rate': 0,
            'is_overheat': False
        }
        
        # 1. 直前セッションOHLC取得
        prev_ohlc = self.get_prev_session_ohlc(session)
        if prev_ohlc is None:
            result['reason'] = 'INSUFFICIENT-DATA: 直前セッションデータなし'
            return result
        
        prev_close = prev_ohlc['close']
        prev_range = prev_ohlc['range']
        
        # 2. RiskGate (ギャップチェック)
        risk_pass, gap_rate = self.check_risk_gate(entry_price, prev_close)
        result['gap_rate'] = gap_rate
        
        if not risk_pass:
            result['reason'] = f'RiskGate FAIL: gap={gap_rate*100:.3f}% >= {Config.G_CUT*100}%'
            return result
        
        # 3. B: 直前セッション方向
        score_b = self.calc_score_b(prev_ohlc)
        result['score_b'] = score_b
        
        # 4. C: モメンタム
        score_c = self.calc_score_c()
        result['score_c'] = score_c
        
        # 5. D: 過熱判定
        avg_range = self.get_avg_range(10)
        score_b_adj = self.apply_overheat_d(score_b, prev_range, avg_range)
        result['score_b_adj'] = score_b_adj
        result['is_overheat'] = (score_b != score_b_adj)
        
        # 6. TotalScore
        total_score = score_b_adj + score_c
        result['total_score'] = total_score
        
        # 7. 判定
        if total_score >= 2:
            result['verdict'] = 'BUY'
            result['reason'] = f'B={score_b_adj:+d} C={score_c:+d} Total={total_score:+d}'
        elif total_score <= -2:
            result['verdict'] = 'SELL'
            result['reason'] = f'B={score_b_adj:+d} C={score_c:+d} Total={total_score:+d}'
        else:
            result['verdict'] = 'NO-TRADE'
            result['reason'] = f'B={score_b_adj:+d} C={score_c:+d} Total={total_score:+d} (条件未達)'
        
        return result


# ============================================================
# メインBot
# ============================================================
class NikkeiBot:
    def __init__(self):
        pass
    
    def determine_session(self) -> str:
        """現在時刻からセッションを判定"""
        now = datetime.now(Config.JST)
        hour = now.hour
        minute = now.minute
        
        # 08:00〜09:00 → DAY判定
        if hour == 8 or (hour == 9 and minute == 0):
            return 'DAY'
        # 16:00〜17:30 → NIGHT判定
        elif (hour == 16) or (hour == 17 and minute <= 30):
            return 'NIGHT'
        else:
            return None
    
    def run(self):
        """メイン実行"""
        print("=" * 60)
        print("🦖 Raptor225 Bot v2.0 (内部プロンプト準拠)")
        now = datetime.now(Config.JST)
        print(f"📅 {now.strftime('%Y-%m-%d %H:%M JST')}")
        print("=" * 60)
        
        # セッション判定
        session = self.determine_session()
        if not session:
            print(f"⏰ 現在時刻 {now.hour}:{now.minute:02d} は対応時間外です")
            print("   DAY判定: 08:00〜09:00")
            print("   NIGHT判定: 16:00〜17:30")
            return
        
        print(f"🎯 {session}セッション判定")
        
        # データ取得
        print("\n📥 データ取得中...")
        market = MarketData.fetch()
        if not market:
            send_line("❌ データ取得失敗")
            return
        
        current = market.get('current_price')
        if current is None:
            send_line("❌ 現在価格取得失敗")
            return
        
        entry_price = tick_round(current)
        print(f"   寄付き予想価格: ¥{entry_price:,}")
        
        # Raptor判定
        print("\n🔎 Raptor判定...")
        engine = RaptorEngine(market)
        signal = engine.get_signal(session, entry_price)
        
        # 結果出力
        print(f"\n【判定結果】")
        print(f"   Session:     {signal['session']}")
        print(f"   Entry:       ¥{signal['entry_price']:,}")
        print(f"   Gap Rate:    {signal['gap_rate']*100:.3f}%")
        print(f"   B (調整前):  {signal['score_b']:+d}")
        print(f"   B (調整後):  {signal['score_b_adj']:+d} {'(過熱抑制)' if signal['is_overheat'] else ''}")
        print(f"   C:           {signal['score_c']:+d}")
        print(f"   TotalScore:  {signal['total_score']:+d}")
        print(f"   Verdict:     {signal['verdict']}")
        print(f"   Reason:      {signal['reason']}")
        
        # LINE通知
        if signal['verdict'] in ['BUY', 'SELL']:
            msg = f"""🦖 Raptor225 {session}
📈 {signal['verdict']}
---
Entry: ¥{entry_price:,}
B={signal['score_b_adj']:+d} C={signal['score_c']:+d}
Gap: {signal['gap_rate']*100:.2f}%
---
Exit: {'15:45' if session == 'DAY' else '06:00'}"""
            send_line(msg)
        else:
            send_line(f"🦖 {session}セッション\n→ {signal['verdict']}\n{signal['reason']}")
        
        print("\n✅ 完了")


# ============================================================
# エントリーポイント
# ============================================================
if __name__ == "__main__":
    bot = NikkeiBot()
    bot.run()
