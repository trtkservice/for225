# 🤖 Nikkei 225 Antigravity Bot (v2.1)

## 1. プロジェクトの目的 (Project Goal)
本プロジェクトは、**日経225** を対象とした、統計的優位性に基づく自動売買システムの構築を目的としています。
「Antigravity（反重力）」という名の通り、市場のノイズ（重力）に逆らい、トレンドの発生源（初動）を的確に捉えて利益を浮上させることを目指します。

現在は **フェーズ1（シャドートレーディング）** として、GitHub Actions上で **A/Bテスト（2つの戦略の競争）** を行い、そのパフォーマンスを競わせています。

将来の **フェーズ2（本番運用）** では、現在のシャドートレーディング環境（GitHub Actions）をそのまま利用し、**OANDA Japan (CFD)** のAPIを通じて完全自動売買を行います。

---

## 2. システムアーキテクチャ (Core Logic)

本システムの中核となる **Antigravity Engine v2.1** は、オブジェクト指向設計で構築された堅牢なPythonアプリケーションです。

### 🧠 Antigravity Engine (分析エンジン)
市場データを物理学的なメタファー（Trend, Momentum, Volatility）で解析し、エントリー判断を行います。

| レイヤー | 名称 (Metaphor) | 使用指標 | 役割 |
|:---:|:---:|:---|:---|
| **Layer 1** | **🌊 Trend (River)** | EMA (20, 50, 200) | 大局的なトレンド方向（上昇/下降）を判定 |
| **Layer 2** | **🚀 Momentum (Wind)** | RSI (14), MACD | 短期的な勢いと、買われすぎ/売られすぎの逆張り判定 |
| **Layer 3** | **⚠️ Volatility (Temp)** | VIX Index | 市場の恐怖指数を監視し、危険な相場（VIX>30）を回避 |

**Gemini AI (Audit)**:
最終的なスコア (`-1.0` ~ `+1.0`) に対し、Google Gemini Pro モデルが「リスク管理マネージャー」として監査を行い、突発的なリスク要因がないかダブルチェックを行います。

### ⚔️ A/Bテスト（戦略比較）
現在、同一のEngineを用いて、資金管理ルールの異なる2つの戦略を並行稼働させています。

1.  **Antigravity (Normal)**:
    *   **重視**: 資金効率と回転率。
    *   **Stop**: 0.6 ATR
    *   **Target**: 1.2 ATR
    *   **特徴**: 小刻みに利確を積み重ねる、バックテストで最高成績を出したモデル。

2.  **Raptor (Wide)**:
    *   **重視**: トレンド追従力（握力）。
    *   **Stop**: 1.2 ATR
    *   **Target**: 2.4 ATR
    *   **特徴**: ノイズに振るい落とされず、大きな波を狙う。プロトレーダーの助言に基づくモデル。

---

## 3. インフラストラクチャ (Infrastructure)

本プロジェクトは **「サーバーレス & PCレス」** を実現しています。
高価なVPSや専用PCは不要です。

### ⚙️ GitHub Actions (Automation)
システムはクラウド上で完全自動運用されています。

1.  **Daily Prediction (`src/nikkei_bot.py`)**:
    *   毎日 **08:00 JST** と **16:00 JST** に定期実行。
    *   市場データの取得 → AI分析 → シグナル判定 → 仮想注文 → 結果記録を一気通貫で行います。
    
2.  **Dashboard Generation**:
    *   取引結果をWebダッシュボード (`dashboard.html`) に即時反映。

### 📊 Dashboard
[View Dashboard](https://trtkservice.github.io/for225/data/dashboard.html)

---

## 4. 本番移行ガイド (Go-Live Strategy)

フェーズ1（シャドートレーディング）で十分な実績が出た後、以下の手順でフェーズ2（本番運用）へ移行します。

### ✅ 推奨環境: OANDA Japan + GitHub Actions
この組み合わせにより、**PCを常時起動する必要も、VPSを借りる必要もありません。**

1.  **口座開設**:
    *   **OANDA Japan** で口座を開設します（コースは「プロコース」等がAPI対応の場合あり、要確認）。
    *   取引銘柄: **JP225 USD** (日経225 CFD)。

2.  **APIアクセストークン発行**:
    *   OANDAの管理画面からAPIトークンを発行します。

3.  **GitHub Actions Secrets 設定**:
    *   GitHubリポジトリの Settings > Secrets に以下を追加します。
        *   `OANDA_API_TOKEN`: (発行したトークン)
        *   `OANDA_ACCOUNT_ID`: (口座ID)

4.  **コードの微修正**:
    *   `src/nikkei_bot.py` の `CONTRACT_MULTIPLIER` を `10` から `1` (または0.1) に変更。
    *   注文処理部分に `requests.post(...)` を追加（OANDA APIを叩く処理）。

これにより、現在の「記録するだけのBot」が「実際に注文を飛ばすBot」に進化します。
追加コストはゼロです。

---

## 5. 開発者向け情報 (Development)

### ディレクトリ構成
```
.
├── src/
│   ├── nikkei_bot.py        # メインロジック (A/B System)
│   ├── generate_dashboard.py # ダッシュボード生成
│   └── __init__.py          # パッケージ定義
├── data/
│   ├── predictions.json     # 取引データDB (JSON)
│   └── dashboard.html       # 生成されたWeb UI
├── backtest_runner.py       # バックテスト実行用スクリプト
└── .github/workflows/       # 自動実行定義
```

### ローカル実行方法
```bash
# 依存ライブラリのインストール
pip install -r requirements.txt

# APIキーの設定
export GEMINI_API_KEY="your_api_key"

# Botの実行 (データ取得〜判定〜記録)
python src/nikkei_bot.py

# ダッシュボード生成
python src/generate_dashboard.py

# バックテスト実行
python backtest_runner.py
```

---
© 2026 Antigravity Project
