# Outputs

このディレクトリは実験成果物を保存します。

## 推奨ルール

- `experiment_name` を必ず指定し、`outputs/<experiment_name>/` に保存する
- 同一 experiment 名で上書きしない（比較が必要な場合は suffix を付ける）

## 生成物

- `submission.csv`
- `oof.csv`
- `cv_scores.json`
- `config_used.json`
- `meta.json`

`meta.json` には環境情報や git commit が含まれます。
