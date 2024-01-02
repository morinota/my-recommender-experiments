## link

- https://recbole.io/docs/developer_guide/customize_metrics.html

# metricsをカスタムする

- カスタマイズした評価指標を実装し、他の評価指標と組み合わせて、モデルの評価に使用できる。
- ステップは3つ:
  - 1. `recbole.evaluator.metrics.AbstractMetric`クラスを継承した新しいクラスを作る。
  - 2. metricクラスの3つの属性を設定する。
    - `metric_need`:
      - metricを計算するために必要な入力情報をstr or list[str]で指定する。
      - https://recbole.io/docs/developer_guide/customize_metrics.html#set-metric-need に現時点で指定可能な入力情報の一覧がある。
        - rec.items: 各ユーザのtopk推薦アイテムの行列。
        - rec.topk: topk推薦アイテムが、testデータにpositive itemsとして存在するか否かのbool行列。
        - rec.meanrank: positive itemsの平均ランク。
        - rec.score: 推薦モデルが出力した各user-itemペアのスコアの行列。
        - data.num_items: dataset内のアイテム数。
        - data.num_users: dataset内のユーザ数。
        - data.count_items: 各アイテムのinteraction数。
        - data.count_users: 各ユーザのinteraction数。
        - data.label: 入力データのlabel。(通常 rec.score と一緒に使われる)
    - `metric_type`:
      - EvaluatorType.RANKING or EvaluatorType.VALUEのどちらかを指定する。
      - metricのスコアがユーザ毎にグループ化されている必要があるか否か。
    - `smaller`:
      - metricのスコアが小さいほど性能が良いかどうかを指定する。(defaultはFalse)
  - 3. `calculate_metric` メソッドを実装する。
    - すべての計算プロセスはこのfunctionの中で定義される。
