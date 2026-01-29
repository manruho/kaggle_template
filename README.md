# Kaggle Template

Kaggle の実験を **「Notebookは薄く」**、**「ロジックは `src/` に閉じ込める」** 方針で回すためのテンプレートです。

- Kaggle Notebook から **GitHub を clone（Internet ON）** して使える
- Internet OFF のコンペでも **Kaggle Dataset に添付**して使える
- Notebook 側は基本 **`load_config` → `run(cfg)` の呼び出しだけ**
- 実験成果物は `outputs/<experiment_name>/` に保存して再現性を担保（`meta/` と `submission/` を生成）

> 目標：
> **「実験を増やす＝configを増やす」**
> **「Notebookは実行ボタン」**
> **「当たり実験を二度と失わない」**

---

## Quick Start (1 Cell)

```python
# clone
!rm -rf /kaggle/working/kaggle-template
!git clone https://github.com/manruho/kaggle_template /kaggle/working/kaggle-template

# import
import sys
sys.path.insert(0, "/kaggle/working/kaggle-template")

from src.config_io import load_config
from src.experiment import run

cfg = load_config("/kaggle/working/kaggle-template/configs/default.json")
result = run(cfg)
result
```

---

## Directory Structure

```
kaggle-template/
  kaggle_notebook_init.py
  NOTEBOOK_TEMPLATE.md
  src/
    config.py
    config_io.py
    data.py
    features.py
    features/
    feature_store.py
    models.py
    models/
    split.py
    train.py
    inference.py
    experiment.py
    tasks/
    utils.py
  configs/
    _base.json
    default.json
    comp_example/
      train.json
  data/
    README.md
  docs/
    README.md
  outputs/
    README.md
  tests/
    test_pipeline.py
    fixtures/
  scripts/
    package_dataset.py
  README.md
```

- **コンペ固有の設定**（データパス、メトリック、CV、モデル）は `configs/*.json` に集約
- `src/` には **汎用的な処理**のみを置く（Notebookにロジックを書かない）

---

## First Things to Change

- `train_path`, `test_path`, `sample_sub_path`
- `id_col`, `target_col`
- `metric`, `cv_method`（推奨）/`cv_type`（互換用）, `n_splits`, `seed`
- `model_name`, `model_params`
- `experiment_name`（`auto` 推奨）
- `feature_version`, `use_feature_cache`
- `save_policy`（`none` / `best` / `all` / `keep_top_k`）
- `save_top_k`, `models_dir`
- `env_packages`, `include_pip_freeze`

---

## Pipeline Overview

`run(cfg)` が司令塔としてパイプラインを実行します（概念図）：

1. `config_io`：JSON / YAML → `Config`（または dict）にロード
2. `data`：学習/推論データの読み込み（`debug` で軽量化）
3. `features`：特徴量生成（必要に応じてダミー化/結合）
4. `split`：CV 分割（kfold / stratified / group / time / repeated）
5. `models`：`model_name` から学習器生成（factory）
6. `train`：CV 学習 → OOF / fold score / test pred
7. `inference`：提出用 DataFrame 作成
8. `experiment` / `utils`：成果物を `outputs/` に保存

Notebook からは **`run(cfg)` 一発**で全処理が起動するため、変更は GitHub 側に集中できます。

---

## 使い方（最短）

`NOTEBOOK_TEMPLATE.md` に最小セル例があります。まずはそれをコピーして開始してください。

---

## 実行モード（分離）

- `scripts/train.py --config <path>`：学習のみ（モデル保存 + OOF）
- `scripts/infer.py --config <path>`：推論のみ（保存済みモデルを使用）
- `scripts/make_submission.py --config <path>`：提出ファイル生成 + 検証

Notebook からも `train_only / infer_only / make_submission` を呼び出せます。

---

## 使い方（GitHub から clone / Internet ON）

1. Notebook の Internet を ON にする。
2. 冒頭でリポジトリを clone し、`sys.path` に追加する。

```python
# clone
!rm -rf /kaggle/working/kaggle-template
!git clone https://github.com/manruho/kaggle_template /kaggle/working/kaggle-template

# import
import sys
sys.path.insert(0, "/kaggle/working/kaggle-template")

from src.config_io import load_config
from src.experiment import run

cfg = load_config("/kaggle/working/kaggle-template/configs/default.json")
result = run(cfg)
result
```

GitHub 上でテンプレートを更新したら、Notebook を再実行（再 clone / pull）して反映します。

`kaggle_notebook_init.py` を使う場合は、clone 後に `sys.path.append(...)` の上で import すると `REPO_DIR` が使えます。

---

## 使い方（Dataset 経由 / Internet OFF）

1. ローカルで `python scripts/package_dataset.py --output kaggle_template_lib.zip` を実行し、`src/` と `configs/` をまとめた zip を作成。
2. Kaggle の Dataset を新規作成し、`kaggle_template_lib.zip` をアップロード。
3. Notebook でその Dataset を「Add data」し、working ディレクトリにコピーして import する。

```python
from pathlib import Path
import shutil
import sys

# ここは Dataset 名に合わせる
LIB_ROOT = Path("/kaggle/input/kaggle-template-lib/kaggle-template")
WORK_ROOT = Path("/kaggle/working/kaggle-template")

if WORK_ROOT.exists():
    shutil.rmtree(WORK_ROOT)
shutil.copytree(LIB_ROOT, WORK_ROOT)

sys.path.insert(0, str(WORK_ROOT))

from src.config_io import load_config
from src.experiment import run

cfg = load_config(str(WORK_ROOT / "configs" / "default.json"))
result = run(cfg)
result
```

Internet が OFF のコンペでも同じワークフローでテンプレートを利用できます。

---

## 一連の運用フロー

1. 新規コンペ用の `configs/comp_xxx/` を作成し、`train.json` などを置く。
2. Notebook から `load_config` → `run(cfg)` を呼び、OOF・提出を生成。
3. 結果は `outputs/<experiment_name>/` に `submission/` と `meta/` を含む成果物を保存。
4. Kaggle の提出や外部分析は保存された成果物を参照するだけで良い。
5. 改良（特徴量やモデル）を行ったら GitHub に commit → push、Notebook は再 clone or Dataset 更新のみ。

---

## Outputs (Experiment Artifacts)

```
outputs/<experiment_name>/
  submission/
    submission.csv
  meta/
    config.snapshot.json
    git.txt
    env.txt
    cv_scores.json
    run_summary.json
  oof.csv
  oof.parquet
  pred_test.parquet
  pred_test.npy
  folds.csv
  models/

outputs/
  experiments.jsonl

## 命名規約（auto）

`<model>_<experiment_version>__fe<feature>__cvX__seedY`

例: `lgb_v1__fe005__cv5__seed42`

## CV 手法

- `kfold`, `stratified`, `group`, `time`, `purged`
- `repeated_kfold`, `repeated_stratified_kfold`
- `cv_params` で `n_repeats` / `purge_gap` などを渡せます
```

* `submission.csv`：提出用
* `oof.csv`：OOF（アンサンブル/stacking用）
* `oof.parquet`：`id`, `target`, `pred`, `fold`
* `pred_test.parquet`：`id`, `pred`
* `cv_scores.json`：fold score
* `meta/config.snapshot.json`：実行時点の config（再現性）
* `meta/cv_scores.json`：CVサマリ（fold別スコア/平均/標準偏差）
* `meta/run_summary.json`：実行サマリ（durationなど）
* `meta/git.txt`：commit/branch/dirty
* `meta/env.txt`：Python/主要ライブラリ（任意でpip freeze）
* `meta/command.txt`：実行コマンド
* `meta/seed.txt`：乱数シード

> ルール：
> **実験結果は `outputs/` を見れば全部わかる**状態を目指します。

---

## Config Design (configs/*.json)

基本キー（例）：

* `train_path`, `test_path`, `sample_sub_path`
* `id_col`, `target_col`
* `task_type`, `metric`
* `cv_type`, `n_splits`, `seed`
* `model_name`, `model_params`
* 追加パラメータ：`extras`（自由枠）

Example:

```json
{
  "train_path": "/kaggle/input/xxx/train.csv",
  "test_path": "/kaggle/input/xxx/test.csv",
  "sample_sub_path": "/kaggle/input/xxx/sample_submission.csv",
  "id_col": "id",
  "target_col": "target",
  "task_type": "binary",
  "metric": "roc_auc",
  "cv_type": "stratified",
  "n_splits": 5,
  "seed": 42,
  "model_name": "lightgbm",
  "model_params": {
    "learning_rate": 0.05,
    "num_leaves": 64
  },
  "extras": {
    "debug": false
  }
}
```

### extras について

`extras` は「このコンペだけ必要な値」を逃がすための自由枠です。例：

* `group_col`（GroupKFold用）
* `use_gpu`（GPUの有無で分岐）
* `features_version`（特徴量版管理）

---

## Feature Engineering

基本は `src/features.py` を編集して特徴量を足します。

```python
def build_feature_frames(train_df, test_df, cfg):
    # 例：特徴量追加（破壊を避けたいなら copy() 推奨）
    X_train = train_df.copy()
    X_test = test_df.copy()

    for df in (X_train, X_test):
        df["amount_per_unit"] = df["amount"] / (df["units"] + 1e-3)
        df["city_country"] = df["city"].astype(str) + "_" + df["country"].astype(str)

    return X_train, X_test
```

---

## Feature Cache (FeatureStore)

同じ特徴量を何度も計算しないために、`feature_store.py` を使ってキャッシュできます。

Config 例：

```json
{
  "use_feature_cache": true,
  "features_version": "v1",
  "feature_cache_dir": "outputs/_cache/features",
  "feature_cache_format": "auto"
}
```

* `features_version`：特徴量ロジックを変えたら更新（事故防止）
* `feature_cache_format`：`auto` / `parquet` / `pickle`

---

## Model Switching / Add Models

### config だけで切り替える

`model_name` を変えるだけで学習器を切り替えられる設計を推奨します。

例：`lightgbm`, `xgboost`, `catboost`, `logistic_regression` など

### 新しいモデルの追加

`src/models.py` に factory を追加し、必要ならライブラリを導入します。

```python
if name == "catboost":
    from catboost import CatBoostClassifier
    return CatBoostClassifier(verbose=0, **params)
```

> Notebook は変えず、configだけで切り替えられるようにするのがポイントです。

---

## Ensemble (Blending / Stacking)

テンプレ本体は「1 config = 1 実験」を守ってシンプルにします。
アンサンブルは Notebook 側で複数実験を回して合成します。

```python
from src.config_io import load_config
from src.experiment import run
import numpy as np

cfg_a = load_config("configs/comp_a.json")
cfg_b = load_config("configs/comp_b.json")

res_a = run(cfg_a)
res_b = run(cfg_b)

pred = 0.6 * res_a.predictions_test + 0.4 * res_b.predictions_test
sub = res_a.submission.copy()
sub["target"] = pred
sub.to_csv("/kaggle/working/submission.csv", index=False)
```

---

## Testing

```bash
pip install -e .[test]
pytest
```

`tests/test_pipeline.py` では、合成データで `run(cfg)` が最後まで動き、
成果物が生成されることを確認します。

---

## Packaging for Kaggle Dataset

Internet OFF 用に zip を作ります：

```bash
python scripts/package_dataset.py --output kaggle_template_lib.zip
```

作成物は `.gitignore` に入れて誤コミットを防ぐのがおすすめです。

---

## Recommended Extensions

* `metric` の種類追加、タスク別 pipeline
* `src/features.py` / `src/models.py` の拡張
* `src/tasks/` を作って tabular / nlp / image を切替
* `meta/git.txt` に commit hash / versions を保存して実験台帳化
* OOF を使った重み最適化 blend（終盤の伸びに効く）

---

## Philosophy

* Notebook は **実行ボタン**
* 実験の差分は **config**
* 成果物は **outputs に全部**
* 当たり実験を **消さない・忘れない・再現できる**
