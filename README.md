# 🤖 Nikkei 225 Antigravity Bot (v2.1)

## 1. プロジェクトの目的 (Project Goal)
本プロジェクトは、**日経225先物（マイクロ）** を対象とした、統計的優位性に基づく自動売買システムの構築を目的としています。
「Antigravity（反重力）」という名の通り、市場のノイズ（重力）に逆らい、トレンドの発生源（初動）を的確に捉えて利益を浮上させることを目指します。

現在は **フェーズ1（シャドートレーディング）** として、GitHub Actions上で仮想売買を行い、そのパフォーマンスを継続的に計測・最適化しています。

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

### 🛡 資金管理とスイングトレード戦略 (Strategy)
v2.1より、日中のノイズを無視し、大きなトレンドを取り切るための**スイングトレード戦略**を採用しました。

1.  **オーバーナイト（夜間持ち越し）解禁**:
    *   従来の「引けで強制決済」を廃止。
    *   **Stop（損切り）** または **Target（利確）** に到達するまで、ポジションを持ち越します。
    *   これにより、夜間（NY市場）の大きな値動きを利益に変えます。

2.  **動的なExitライン (ATRベース)**:
    *   市場のボラティリティ（ATR）に応じて、利確・損切り幅を毎日自動調整します。
    *   **Stop**: ATR × **0.6**
    *   **Target**: ATR × **1.2**（リスクリワード 1:2）
    
3.  **期限設定**:
    *   資金効率を考慮し、最大 **5日間** 経過しても決着がつかない場合はタイムアウトとして決済します。

---

## 3. インフラストラクチャ

### ⚙️ GitHub Actions (Automation)
システムはクラウド上で完全自動運用されています。

1.  **Daily Prediction (`src/nikkei_bot.py`)**:
    *   毎日 **16:00 JST** に定期実行。
    *   市場データの取得 → AI分析 → シグナル判定 → 仮想注文 → 結果記録を一気通貫で行います。
    
2.  **Ad-hoc Backtest (`backtest_runner.py`)**:
    *   任意のタイミングで過去3年分のバックテストを実行可能。
    *   最新のロジックが過去の相場で通用するかを即座に検証できます。

### 📊 Dashboard
運用成績は、自動生成されるHTMLダッシュボードで可視化されます。
*   **KPI**: 資産推移曲線、勝率、プロフィットファクター、直近のトレード履歴
*   [View Dashboard](https://trtkservice.github.io/for225/data/dashboard.html)

---

## 4. 運用実績 (Verification)

### 最新バックテスト結果 (2023-2026)
*   期間: 直近3年間
*   ロジック: Antigravity v2.1 (Swing Mode)
*   **Total Return: +100.94%** (資産倍増)
*   **Win Rate: 43.2%**
*   **Risk Reward**: 1 : 2

---

## 5. 開発者向け情報 (Development)

### ディレクトリ構成
```
.
├── src/
│   ├── nikkei_bot.py        # メインロジック (Class-based)
│   └── generate_dashboard.py # ダッシュボード生成
├── data/
│   ├── predictions.json     # 取引データDB
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
