# Outputs

このディレクトリは実験成果物を保存します。

## 推奨ルール

- `experiment_name` は `auto` を推奨（規約化された名前が生成される）
- 同一 experiment 名で上書きしない（比較が必要な場合は suffix を付ける）

## 生成物

- `submission/submission.csv`
- `meta/config.snapshot.json`
- `meta/git.txt`
- `meta/env.txt`
- `meta/cv_scores.json`
- `meta/run_summary.json`
- `oof.csv`
- `oof.parquet`
- `pred_test.parquet`
- `pred_test.npy`
- `folds.csv`
- `models/`

`outputs/experiments.jsonl` に各実験のサマリが追記されます。
