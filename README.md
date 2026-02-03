# 🤖 Nikkei 225 AI Trading Bot

日経225先物の売買シグナルを自動生成するボットです。

## 機能

- **自動データ取得**: 日経225、米国株、VIX、ドル円などを自動取得
- **AI判定**: Gemini 1.5 Pro による売買シグナル生成
- **シャドートレーディング**: 仮想10万円でのパフォーマンス記録
- **完全自動実行**: GitHub Actions で毎日8時・16時に自動実行

## セットアップ

1. このリポジトリをfork

2. Settings → Secrets → Actions で以下を追加:
   - `GEMINI_API_KEY`: Google AI Studio で取得したAPIキー

3. Actions タブで「Daily Nikkei Prediction」を有効化

## 使い方

### 自動実行
毎日 8:00 JST と 16:00 JST に自動で実行されます。

### 手動実行
Actions → Daily Nikkei Prediction → Run workflow

### 結果確認
`data/predictions.json` に全ての予測とシャドートレード結果が保存されます。

## シャドーポートフォリオ

- 初期資金: ¥100,000
- ポジションサイズ: マイクロ先物1枚
- 1ポイント = ¥10

## ファイル構成

```
├── .github/workflows/
│   └── daily_prediction.yml  # 自動実行設定
├── src/
│   └── nikkei_bot.py         # メインスクリプト
├── data/
│   └── predictions.json      # 予測・取引記録
└── requirements.txt          # 依存ライブラリ
```

## 注意事項

⚠️ このボットはテスト・研究目的です。実際の取引は自己責任で行ってください。
