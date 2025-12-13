# Kaggle Template

Kaggle ノートブックを「GitHub から直接 clone して使う」「Kaggle Dataset として添付して使う」のどちらでも再利用できるテンプレートです。実験ロジックは `src/` 以下に閉じ込め、Notebook 側は `load_config` と `run(cfg)` を呼ぶだけで済む構成になっています。

## ディレクトリ構成

```
kaggle-template/
  src/
    config.py
    config_io.py
    data.py
    features.py
    models.py
    split.py
    train.py
    inference.py
    experiment.py
    utils.py
  configs/
    default.json
  tests/
    test_pipeline.py
  scripts/
    package_dataset.py
  README.md
```

各コンペ固有の設定値（データのパスやメトリックなど）は `configs/*.json` に集約します。Python コードは汎用的な処理のみを保持します。

### 役割とデータフロー

1. `src/config_io.py` で JSON → `Config` オブジェクトへ変換。
2. `src/experiment.py` の `run(cfg)` がパイプラインを司令塔として実行。
3. `src/data.py` で CSV を読み込み、`debug` オプションで試行を軽量化。
4. `src/features.py` で使用する列を選択し、カテゴリのダミー化や結合を実施。
5. `src/split.py` で CV の分割方法（KFold/Stratified/Group）を決定。
6. `src/models.py` が `model_name` から学習器を生成。
7. `src/train.py` で `train_cv` を実行し、OOF・CV スコア・テスト予測を得る。
8. `src/utils.py` と `experiment.py` が提出ファイル、OOF、スコアログ、使用した config を `outputs/` に保存。

このように Notebook からは `run(cfg)` 一発で全処理が起動するため、コード本体の変更は GitHub 側に集中させられます。

## 使い方（GitHub から clone / Internet ON）

1. Notebook の Internet を ON にする。
2. 冒頭でリポジトリを clone し、`sys.path` に追加する。

```python
!git clone https://github.com/<you>/kaggle-template.git
import sys
sys.path.append("/kaggle/working/kaggle-template")

from src.config_io import load_config
from src.experiment import run

cfg = load_config("/kaggle/working/kaggle-template/configs/default.json")
result = run(cfg)
```

GitHub 上でテンプレートを更新すると、clone している Notebook 全てに自動で反映されます。

## 使い方（Dataset 経由 / Internet OFF）

1. ローカルで `python scripts/package_dataset.py --output kaggle_template_lib.zip` を実行し、`src/` と `configs/` をまとめた zip を作成。
2. Kaggle の Dataset を新規作成し、zip をアップロード。
3. Notebook でその Dataset を「Add data」し、working ディレクトリにコピーして import する。

```python
from pathlib import Path
import shutil, sys

LIB_ROOT = Path("/kaggle/input/kaggle-template-lib")  # 自分の Dataset 名に合わせる
WORK_ROOT = Path("/kaggle/working/kaggle-template")

if not WORK_ROOT.exists():
    shutil.copytree(LIB_ROOT, WORK_ROOT)

sys.path.append(str(WORK_ROOT))

from src.config_io import load_config
from src.experiment import run

cfg = load_config(str(WORK_ROOT / "configs" / "default.json"))
result = run(cfg)
```

Internet が OFF のコンペでも同じワークフローでテンプレートを利用できます。

## 一連の運用フロー

1. 新規コンペ用の `configs/comp_xxx.json` を作成。
2. Notebook から `load_config` → `run(cfg)` を呼び、OOF・提出を生成。
3. 結果は `outputs/<experiment_name>/` に `submission.csv`, `oof.csv`, `cv_scores.json`, `config_used.json` として保存。
4. Kaggle の提出や外部分析は保存された成果物を参照するだけで良い。
5. 改良（特徴量やモデル）を行ったら GitHub に commit → push、Notebook は再 clone or Dataset 更新のみ。

## コンフィグ（configs/*.json）の考え方

- `train_path`, `test_path`, `sample_sub_path`
- `id_col`, `target_col`
- `task_type`, `metric`, `cv_type`, `n_splits`, `seed`
- `model_name`, `model_params`
- 追加で必要な値は `extras` として自由に定義

例:

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
  }
}
```

コンペ切り替え時は JSON だけ差し替えれば Notebook の修正を最小化できます。

### 付加情報を渡す

`Config` には `extras` という自由枠があります。`configs/*.json` に `group_col`, `use_gpu` など任意のキーを追加すると `config.get("group_col")` のように取得できます。分割にグループ列を使う、特徴量処理を切り替える、といった条件分岐に便利です。

## 特徴量エンジニアリングを追加する

`src/features.py` を編集して処理を差し込むのが基本です。

```python
# 例: 新しいカテゴリ集約特徴を追加
def build_feature_frames(train_df, test_df, config):
    # 既存の列選択
    ...
    # 追加の特徴量
    for df in (X_train, X_test):
        df["amount_per_unit"] = df["amount"] / (df["units"] + 1e-3)
        df["city_country"] = df["city"] + "_" + df["country"]
    return _align_dummies(X_train, X_test)
```

ポイント:

- 元の `train_df` / `test_df` を破壊したくない場合は `copy()` してから加工。
- 設定ファイルで `features` リストを与えると、使用する列を固定できます。自動選択したい場合は `drop_cols` に不要列を入れるだけ。
- 高度な処理が必要になったら `src/features/` 配下にモジュールを増やし、`build_feature_frames` から呼び出す形に整理すると拡張しやすいです。

## モデルを変更・追加する

### 設定だけで切り替える

`configs/*.json` の `model_name` を `logistic_regression` → `lightgbm` などに変えるだけで、`src/models.py` のファクトリが対応モデルを返します。`model_params` で LightGBM や XGBoost のハイパーパラメータも渡せます。

### 新しいモデルクラスを追加する

1. `src/models.py` に分岐を追加。
2. 必要なライブラリを `pyproject.toml`（または Kaggle Notebook 側の `pip install`）で導入。

```python
if name == "catboost":
    from catboost import CatBoostClassifier
    return CatBoostClassifier(verbose=0, **params)
```

Notebook から何も変更せずに config のみで切り替えられるよう意識するのがポイントです。

## アンサンブルを回す

テンプレ本体は 1 実験 = 1 `Config` の思想でシンプルさを維持します。アンサンブルや stacking は Notebook 側で複数の config を読み込み、`run(cfg)` を複数回呼び出して合成します。

```python
from src.config_io import load_config
from src.experiment import run
import numpy as np

cfg_lgb = load_config("configs/comp_lgb.json")
cfg_xgb = load_config("configs/comp_xgb.json")

res_lgb = run(cfg_lgb)
res_xgb = run(cfg_xgb)

ensemble_pred = 0.6 * res_lgb.predictions_test + 0.4 * res_xgb.predictions_test
submission = res_lgb.submission.copy()
submission["target"] = ensemble_pred
submission.to_csv("/kaggle/working/submission.csv", index=False)
```

OOF も `res_lgb.oof` などから取得できるため、外部で stacking モデルを学習させることも容易です。

## 実験管理

- `Config.output_dir` と `experiment_name` を設定すると `outputs/<experiment_name>/` の直下に成果物がまとまります。
- `config_used.json` は実行時点の設定を必ずコピーするので、後から「どのパラメータで回したか」を追跡できます。
- `cv_scores.json` で fold ごとのスコアを確認可能。
- `oof.csv` には `id` + `oof_target` を書き出しており、外部メタモデルやブレンディングに流用できます。
- 追加でメトリクスやログを残したい場合は `experiment.py` の `_write_artifacts` を拡張してください（例: `feature_importance.csv` や `training_log.json` を保存）。

## テスト

```bash
cd kaggle-template
pip install -e .[test]
pytest
```

`tests/test_pipeline.py` では合成データを生成し、`run(cfg)` が学習・OOF 出力・提出ファイルの作成まで行えることを確認します。

## Dataset 用パッケージング

`scripts/package_dataset.py` を実行すると、`src/` と `configs/`（および README）を 1 つの zip にまとめます。Kaggle Dataset にアップロードしておくと、Internet OFF の Notebook からも簡単に利用できます。

```
python scripts/package_dataset.py --output kaggle_template_lib.zip
```

作成された zip は `.gitignore` に含めており、誤ってコミットされません。

## よくある拡張

- `metric` の種類を増やす、タスク別の pipeline を追加する
- `src/features.py` / `src/models.py` に特徴量生成・モデルスイッチを追加
- 画像や NLP など用途別に `src/tasks/` で切り替える

まずは骨格としてこのテンプレートを育て、GitHub で履歴管理しつつ Kaggle Notebook では `run(cfg)` を呼ぶだけの運用を目指してください。
