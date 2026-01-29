# Outputs

このディレクトリは実験成果物を保存します。

## 推奨ルール

- `experiment_name` は `auto` を推奨（規約化された名前が生成される）
- 同一 experiment 名で上書きしない（比較が必要な場合は suffix を付ける）

## 生成物

- `submission.csv`
- `oof.csv`
- `oof.parquet`
- `pred_test.parquet`
- `pred_test.npy`
- `cv_scores.json`
- `config_used.json`
- `run_summary.json`
- `env.txt`
- `pip_freeze.txt`
- `folds.csv`
- `submission_validation.json`
- `models/`
- `meta.json`

`meta.json` には環境情報や git commit が含まれます。

`outputs/experiments.csv` に各実験のサマリが追記されます。
