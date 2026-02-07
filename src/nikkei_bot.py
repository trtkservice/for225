#!/usr/bin/env python3
"""
Raptor225 Trading Bot
=====================
楽天証券 / 資金10万円 / 日経225マイクロ1枚 / デイトレード

GitHub Actionsから毎日実行され、シグナルをLINEに通知する。
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
    COMMISSION = 22        # 往復手数料
    TICK = 5               # 呼値
    
    # Raptorロジック
    GAP_CUT = 0.0025       # ギャップ閾値 0.25%
    STOP_MULT = 1.0        # ストップ = 1.0 ATR
    TARGET_MULT = 2.0      # ターゲット = 2.0 ATR
    
    # データファイル
    DATA_FILE = "data/portfolio.json"
    
    # ティッカー
    TICKER = "NKD=F"       # CME日経225先物

# ============================================================
# ユーティリティ
# ============================================================
def tick_round(price):
    """5円刻みに丸める"""
    return int(round(price / Config.TICK) * Config.TICK)

def calc_slope(closes):
    """終値配列の回帰傾き"""
    n = len(closes)
    if n < 2:
        return 0
    x = np.arange(n)
    y = closes.values if hasattr(closes, 'values') else closes
    return np.polyfit(x, y, 1)[0]

def send_line(message):
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
            
            # 日足 (ATR計算用)
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
# ポートフォリオ管理
# ============================================================
class Portfolio:
    def __init__(self):
        self.data = self._load()
    
    def _load(self):
        """JSONからロード"""
        path = Path(Config.DATA_FILE)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {
            'capital': Config.CAPITAL,
            'position': None,
            'trades': [],
            'predictions': []
        }
    
    def save(self):
        """JSONに保存"""
        path = Path(Config.DATA_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def has_position(self):
        return self.data.get('position') is not None
    
    def open_position(self, action, entry, stop, target):
        """ポジション開始"""
        self.data['position'] = {
            'action': action,
            'entry': entry,
            'stop': stop,
            'target': target,
            'opened_at': datetime.now(Config.JST).isoformat()
        }
        self.save()
    
    def close_position(self, exit_price, reason):
        """ポジション決済"""
        pos = self.data['position']
        if not pos:
            return None
        
        entry = pos['entry']
        action = pos['action']
        
        if action == 'BUY':
            diff = exit_price - entry
        else:
            diff = entry - exit_price
        
        pnl = diff * Config.MULTIPLIER * Config.LOTS - Config.COMMISSION
        
        trade = {
            'action': action,
            'entry': entry,
            'exit': exit_price,
            'pnl': pnl,
            'reason': reason,
            'closed_at': datetime.now(Config.JST).isoformat()
        }
        
        self.data['trades'].append(trade)
        self.data['capital'] += pnl
        self.data['position'] = None
        self.save()
        
        return trade
    
    def log_prediction(self, session, action, entry, stop, target):
        """予測ログ"""
        self.data['predictions'].append({
            'date': datetime.now(Config.JST).strftime('%Y-%m-%d'),
            'session': session,
            'action': action,
            'entry': entry,
            'stop': stop,
            'target': target
        })
        self.save()

# ============================================================
# Raptorシグナル
# ============================================================
class RaptorEngine:
    def __init__(self, market_data):
        self.data = market_data
    
    def get_signal(self, session):
        """
        セッションに応じたシグナルを生成
        
        Args:
            session: 'DAY' or 'NIGHT'
        
        Returns:
            dict: {action, entry, stop, target} or None
        """
        daily = self.data.get('daily')
        intraday = self.data.get('intraday')
        current = self.data.get('current_price')
        
        if daily is None or intraday is None or current is None:
            return None
        
        if len(daily) < 2 or len(intraday) < 10:
            return None
        
        # 直前セッションのトレンド (日足の最後のバーで代用)
        prev_open = daily.iloc[-2]['Open']
        prev_close = daily.iloc[-2]['Close']
        
        # B判定
        if prev_close > prev_open:
            score_b = 1
        elif prev_close < prev_open:
            score_b = -1
        else:
            score_b = 0
        
        # C判定 (15分足の傾き)
        slope = calc_slope(intraday['Close'].iloc[-48:])
        score_c = 1 if slope > 0 else -1
        
        # 合計スコア
        total = score_b + score_c
        
        if total >= 2:
            action = 'BUY'
        elif total <= -2:
            action = 'SELL'
        else:
            return None
        
        # ギャップチェック
        gap = abs(current - prev_close) / prev_close
        if gap >= Config.GAP_CUT:
            print(f"⚠️ ギャップ {gap*100:.2f}% >= {Config.GAP_CUT*100}% → 見送り")
            return None
        
        # ATR計算
        daily['Range'] = daily['High'] - daily['Low']
        atr = daily['Range'].rolling(14).mean().iloc[-1]
        if pd.isna(atr) or atr <= 0:
            atr = 400
        
        entry = tick_round(current)
        s_dist = tick_round(atr * Config.STOP_MULT)
        t_dist = tick_round(atr * Config.TARGET_MULT)
        
        if action == 'BUY':
            stop = entry - s_dist
            target = entry + t_dist
        else:
            stop = entry + s_dist
            target = entry - t_dist
        
        return {
            'action': action,
            'entry': entry,
            'stop': stop,
            'target': target,
            'score_b': score_b,
            'score_c': score_c,
            'atr': atr
        }

# ============================================================
# メインBot
# ============================================================
class NikkeiBot:
    def __init__(self):
        self.portfolio = Portfolio()
    
    def determine_session(self):
        """現在時刻からセッションを判定"""
        now = datetime.now(Config.JST)
        hour = now.hour
        
        # 08:00〜09:00 → DAY判定
        if 8 <= hour < 9:
            return 'DAY'
        # 16:00〜17:30 → NIGHT判定
        elif 16 <= hour < 18:
            return 'NIGHT'
        else:
            return None
    
    def run(self):
        """メイン実行"""
        print("=" * 60)
        print("🦖 Raptor225 Bot v1.0")
        now = datetime.now(Config.JST)
        print(f"📅 {now.strftime('%Y-%m-%d %H:%M JST')}")
        print("=" * 60)
        
        # セッション判定
        session = self.determine_session()
        if not session:
            print(f"⏰ 現在時刻 {now.hour}:{now.minute:02d} は判定時間外です")
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
        print(f"   現在価格: ¥{current:,.0f}")
        
        # シグナル生成
        print("\n🔎 シグナル判定...")
        engine = RaptorEngine(market)
        signal = engine.get_signal(session)
        
        if not signal:
            print("   → NO-TRADE (シグナル条件未達)")
            send_line(f"🦖 {session}セッション\n→ NO-TRADE")
            return
        
        action = signal['action']
        entry = signal['entry']
        stop = signal['stop']
        target = signal['target']
        
        print(f"   B={signal['score_b']:+d} C={signal['score_c']:+d}")
        print(f"   → {action}")
        print(f"   Entry: ¥{entry:,}")
        print(f"   Stop:  ¥{stop:,}")
        print(f"   Target:¥{target:,}")
        
        # ログ保存
        self.portfolio.log_prediction(session, action, entry, stop, target)
        
        # LINE通知
        msg = f"""🦖 {session}セッション
📈 {action}
---
Entry: ¥{entry:,}
Stop:  ¥{stop:,} (損切り)
Target:¥{target:,} (利確)
---
ATR: {signal['atr']:.0f}"""
        send_line(msg)
        
        print("\n✅ 完了")

# ============================================================
# エントリーポイント
# ============================================================
if __name__ == "__main__":
    bot = NikkeiBot()
    bot.run()
