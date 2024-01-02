# Samplerクラス

- RecBoleでは、**訓練や評価に用いるnegative itemsを選択する**ためのサンプラーモジュールが設計されている。

# Samplerをカスタムする。

- 複雑なサンプリング手法が必要な場合にSamplerクラスをカスタムする。
  - ちなみに現在、RecBoleがサポートしているサンプリング戦略は以下の2つのみ:
    - randomネガティブサンプリング
    - popularネガティブサンプリング
- AbstractSamplerクラスを継承して、必要に応じて以下のメソッドをオーバーライドする。
  - `__init__()`メソッド
  - `uni_sampling()`メソッド
  - `_get_candidate_list()`メソッド
  - `get_user_ids()`メソッド
