# 🤖 Nikkei 225 Antigravity Bot

## 1. プロジェクトの目的 (Project Goal)
このシステムは、**楽天証券の日経225先物（マイクロ）** を対象とした完全自動売買システムの構築を目指すプロジェクトです。

現在は**フェーズ1（シャドートレーディング）** として、実際の資金を使わずに仮想売買を行い、AI×クオンツロジックの勝率と収益性を検証しています。

将来的な**フェーズ2（本番運用）** では、VPS上でSeleniumを使用し、楽天証券の管理画面を自動操作して実際の注文を行うことを目指します。

## 2. システムの仕組み (Core Logic)

毎朝8:00と夕方16:00に、GitHub Actions上で以下のプロセスが全自動で行われます。

### A. 分析エンジン "Antigravity Engine v2.0"
メインスクリプト: `src/nikkei_bot.py`

売買の判断（方向）は、以下の定量的・統計的な分析に基づいて決定されます。

1. **Pythonによるマルチレイヤー分析**:
   - **🌊 Trend Layer (DEEP)**: 日足EMA（20, 50, 200）の配列分析により、長期トレンドの方向と強さを数値化。
   - **🚀 Momentum Layer (FAST)**: 15分足のRSI, MACDを使用し、短期的な勢いと過熱感を数値化。
   - **⚠️ Volatility Layer (RISK)**: VIX指数から市場の恐怖度を測定し、エントリーのリスクを評価。
   - **統合スコア**: これらを加重平均し、`-1.0` (強い売り) 〜 `+1.0` (強い買い) の最終スコアを算出します。

2. **Gemini AIによる最終確認**:
   Geminiは意思決定者ではなく**「リスク管理アドバイザー」**として機能します。Pythonが算出したスコアを読み込み、論理的な矛盾や突発的なリスク要因（ブラックスワン）がないかをチェックし、承認（Approved）を行います。

### B. 資金管理 (Risk Management)
感情を排したトレードを行うため、利確・損切りの幅も数式で決定します。

- **ATR (Average True Range)** を使用し、その日のボラティリティに合わせて幅を動的に調整。
- **Stop (損切)**: ATR × 0.5
- **Target (利確)**: ATR × 1.0
- **リスクリワード比**: 常に **1:2** を維持し、統計的優位性を確保。

## 3. 実行環境とフェーズ

### フェーズ1：シャドートレーディング（現在）
- **インフラ**: GitHub Actions (Ubuntu Linux)
- **スケジュール**: 
  - **08:00 (JST)**: デイセッション（日中）の予想＆仮想注文
  - **16:00 (JST)**: ナイトセッション（夜間）の予想＆仮想注文
- **シミュレーション精度**:
  - 日中/夜間の `High` / `Low` （高値・安値）データを使用し、価格到達を判定。
  - **コンサバ判定**: 同日にStopとTargetの両方に到達した場合、**「Stop（負け）」**として記録する厳格なルールを採用。

### フェーズ2：本番運用（将来）
- **インフラ**: VPS (Windows or Linux)
- **注文方法**: Seleniumによるブラウザ自動操作で「IFO注文（イフダン・オーシーオー）」を発注。

## 4. 運用ルール：セッション分離型

本システムは**「持ち越し（オーバーナイト）禁止」**を徹底しています。

- **デイセッション**: 15:15の大引けで必ず強制決済。
- **ナイトセッション**: 翌朝06:00の引けで必ず強制決済。

これにより、NY市場等の急変による「窓開け大損」リスクを物理的に回避し、DayとNightそれぞれのトレンドを独立して収益化します。

## 5. Dashboard
最新の予測と成績は、GitHub Pagesで生成されるダッシュボードで確認できます。
[View Dashboard](https://trtkservice.github.io/for225/data/dashboard.html)
※ リポジトリがPrivateの場合は、ローカルで確認してください。

## 6. Development

### Setup
```bash
pip install -r requirements.txt
```

### Run Locally
```bash
# Set your API Key
export GEMINI_API_KEY="your_api_key"

# Run Bot
python src/nikkei_bot.py

# Generate HTML Dashboard
python src/generate_dashboard.py
```
