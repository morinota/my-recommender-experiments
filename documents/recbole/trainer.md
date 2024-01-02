## link

- https://recbole.io/docs/developer_guide/customize_trainers.html

# Trainer

- AbstractTrainerクラス:
  - 推薦モデルの学習と評価を管理する抽象クラス。Trainerクラスの基底クラス。
  - 引数:
    - config: learning_rate、epochs、eval_stepなど、学習と評価を制御するためのパラメータ情報を含む。
    - model: 推薦モデル。
  - メソッド:
    - fit(train_data): 学習を行う。
    - evaluate(eval_data): 評価を行う。
    - set_reduce_hook(): 分散学習時に使用する?
    - sync_grad_loss():
- Trainerクラス:
  - 基本的な学習・評価戦略のための基本クラス。 ほとんどの推薦システムモデルの学習や評価プロセスのための共通関数を定義している。
  - **単一の損失関数を最適化するケースでは、ほぼこのクラスで問題ない**...!
  - **後述するクラスは全てTrainクラスを継承している**
  - メソッド:
    - `fit(train_data:DataLoader, valid_data:DataLoader=None, verbose=True, saved=True, show_progress=False, callback_fn=None):->(float, dict)`:
      - 学習データと検証データを用いてモデルを学習する。
      - 引数:
        - verbose: ログを出力するかどうか。
        - saved: model parameterを保存するかどうか。
        - show_progress: 学習・検証の進捗を表示するかどうか。
        - callback_fn: epoch終了時に実行されるオプションのcallback関数。
      - 返り値:
        - 検証用データに対するbest scoreとbest result。検証用データを指定しない場合は(-1,None)を返す。
    - `evaluate(eval_data:Dataloader, load_best_model=True, model_file=None, show_progress=False):->dict`:
      - 評価データを用いてモデルを評価する。
      - 引数:
        - eval_data: 評価データ。
        - load_best_model: 学習時のbest modelを使って評価するかどうか。学習直後に評価する場合はTrueを指定する。
        - model_file: もしユーザが以前に学習したモデルファイルをテストしたい場合、このパラメータを設定する。
        - show_progress: 評価の進捗を表示するかどうか。
      - 返り値:
        - key: metric名、value: metric scoreの辞書。計算するmetricはconfigで指定したやつ。
    - `resume_checkpoint(resume_file:file):->None`:
      - 以前に保存したモデルのcheckpointをロードする。
      - 引数:
        - resume_file: checkpointファイルのパス。
- DecisionTreeTrainerクラス:
- KGATTrainerクラス:
- KGTrainerクラス:
- LightGBMTrainerクラス:
- MKRTrainerクラス:
- NCLTrainerクラス:

# Trainerをカスタムする場合

- Trainerクラスを継承して、学習や評価のためのメソッドを適宜オーバーライドする。
- 学習時`fit()`メソッドでは、`_train_epoch()`メソッドを呼び出しているので、独自の学習ロジックを実装する場合はこの辺りをオーバーライドする。
- 評価時`evaluate()`メソッドでは、`_valid_epoch()`メソッドを呼び出しているので、独自の評価ロジックを実装する場合はこの辺りをオーバーライドする。
- カスタムする例:
  - Alternative Optimization (複数の損失関数を交互に最適化したい場合)
  - Mixed precision training (16 ビットと 32 ビット浮動小数点型の両方を使ってモデルのトレーニングを高速化し、使用するメモリを少なくする手法の場合)
