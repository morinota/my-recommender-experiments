- https://recbole.io/docs/user_guide/config_settings.html

# Config Introduction

- RecBoleは実験設定を制御する為に、異なるhyper parametersを設定できる。
  - ex. 前処理方法、分割方法、学習方法、評価方法。

## Environment Settings

- 実行環境の基本的なparametersをデザインできる。
  - ex. GPUのid、workerの数,etc.
  - 各parametersは https://recbole.io/docs/user_guide/config/environment_settings.html
-

## Data settings

- Atomic File Formatに関するconfigs
- Basic Informationに関するconfigs

## Training Settings

- モデルのトレーニングに関するパラメーターを設定するためのものである。
- 参考: https://recbole.io/docs/user_guide/config/training_settings.html

## Evaluation Settings

- モデルの評価に関するパラメータを設定するためのもの。
- https://recbole.io/docs/user_guide/config/evaluation_settings.html

# How to Set config

- RecBoleには三種類のconfig設定方法がある:
  - config files
  - parameter dicts
  - command line 引数で渡す
- 渡された設定は、`recbole.config`モジュールに割り当てられる。

## 方法1 config files

- コンフィグファイルはyamlのフォーマットで整理されるべき。
  - ユーザーはyamlに沿ったルールに従ってパラメータを記述する。
  - 最終的なコンフィグファイルは `recbole.config` モジュールによって処理され、パラメータ設定が完了する。
  - `config_file_list`引数は複数のyamlファイルに対応している。
- Dataset parametersと基本パラメータのEvaluation Settingsに属するパラメータは、設定を再利用するのに便利なconfigファイルに記述することが推奨される。

```yaml
gpu_id: 1
training_batch_size: 1024
```

```python
from recbole.config import Config

config = Config(model='BPR', dataset='ml-100k', config_file_list=['example.yaml'])
print('gpu_id: ', config['gpu_id'])
print('training_batch_size: ', config['training_batch_size'])
```

## 方法2 parameter dicts

- Parameter Dictはpythonのdictデータ構造で実現され、keyはパラメータ名、valueはパラメータ値として指定する。
  - ユーザはパラメータをdictに記述し、configモジュールに入力することができる。

```python
from recbole.config import Config

parameter_dict = {
    'gpu_id': 2,
    'training_batch_size': 512
}
config = Config(model='BPR', dataset='ml-100k', config_dict=parameter_dict)
print('gpu_id: ', config['gpu_id'])
print('training_batch_size: ', config['training_batch_size'])
```

## 方法3 command line 引数で渡す

- コマンドライン上でパラメータを割り当てることもできる。
  - configモジュールは、コマンドラインのパラメータを読み込むことができる。
  - 書式は以下の通り： `--parameter_name=[parameter_value]`

```python
from recbole.config import Config

config = Config(model='BPR', dataset='ml-100k')
print('gpu_id: ', config['gpu_id'])
print('training_batch_size: ', config['training_batch_size'])
```

```
python run.py --gpu_id=3 --training_batch_size=256
```

## 各方法の優先順位:

- RecBoleは3種類のパラメータ設定を組み合わせる事をサポートしている。
  - 異なる設定方法のconfigに重複が合った場合、優先順位は `コマンドライン > パラメータ辞書 > 設定ファイル > デフォルト設定`の順で適用される。
