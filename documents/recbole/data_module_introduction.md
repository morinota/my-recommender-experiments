## link

- https://recbole.io/docs/user_guide/data_intro.html

# Data Module Introduction

- RecBoleは多くの一般的な推薦モデルを実装しているだけでなく、よく使われる28の公開データセットを収集し、公開している。
  - 各データセットは、[データセットダウンロード](https://recbole.io/docs/user_guide/data/dataset_download.html)のドキュメントに従って自由にダウンロードできる。
  - ダウンロード方法は３つ。
- 拡張性と再利用性のために、Recboleには柔軟で拡張可能な`data`モジュールがある。
  - Recboleの`data`モジュールは、生データをモデル入力に変換するエレガントなデータフローを設計する。
  - データフローとして**様々な推薦タスクが必要とする入力データのほとんどの形式を共通化する**ために、RecBoleは**アトミックファイルと呼ばれる入力データ形式**を設計する。
  - **すべての入力データはアトミックファイル形式に変換されなければならない**。
  - また、異なる推薦アルゴリズムに統一的な内部データ表現を提供するために、 `Interaction` と呼ばれるデータ構造を設計する。
- RecBoleは、explicit feedback(ラベル付きデータ)とimplicit feedback(ラベルなしデータ)の両方をサポートする。
  - explicit feedbackを扱う場合、開発者はconfigで`LABEL_FIELD`を設定することができ、RecBoleはそのラベルに基づいてモデルの訓練とテストを行う。
  - implicit feedbackを扱う場合、RecBoleは観測された全ての相互作用をpositiveサンプルとみなし、観測されていない相互作用から自動的にnegativeサンプルを選択する。
    - (これは負のサンプリングとして知られている。)
    - RecBoleのラベル設定についての詳細は、[Label of data](https://recbole.io/docs/user_guide/data/label_of_data.html)をお読みください。

## Dataset Download

- RecBoleは多くの一般的な推薦モデルを実装しているだけでなく、よく使われる28の公開データセットを収集し、公開している。
  - 各データセットは、[データセットダウンロード](https://recbole.io/docs/user_guide/data/dataset_download.html)のドキュメントに従って自由にダウンロードできる。
  - ダウンロード方法は以下の3つ。

### 方法1: Automatiacally downloading

- RecBoleでは、利便性向上のため、**自動ダウンロードモジュール**を実装し、一般に公開されている28のデータセットのアトミックファイルをダウンロードできるようにした。
  - データセットを指定してモデルを実行したい場合、データセットを設定するだけで、自動的に対象のアトミックファイルがダウンロードされる。
  - 例えば、ml-1mデータセットでBPRモデルを実行したいが、ml-1mデータセットのAtomic Filesを用意していない場合、自動ダウンロードモジュールを使ってデータをダウンロードすることができる。
    - RecBoleは自動的にデータファイルがあるかどうかをチェックし、なければダウンロードを開始する。
- これは特に何もconfigを指定する必要はなく、おそらくdataset名さえRecBoleが指定している名前にすれば良い。ex. `ml-100k`
  - "RecBoleが指定しているdataset名"がどこに載ってるか。
  - 他にも困っている人がいたみたい: https://github.com/RUCAIBox/RecBole/discussions/792
  - うーん、上の質問回答を読んだ感じだと、結局のところ、手動でデータセットを準備しないといけないのかな。

### 方法2: クラウドdeskから手動でダウンロード

- 自動ダウンロードの他に、[Google Drive](https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj?usp=sharing)と[Baidu Wangpan](https://pan.baidu.com/s/1p51sWMgVFbAaHQmL4aD_-g) (Password: e272)に、収集・変換した28データセットのatomicファイルをアップロードしている。
- この2つのリソースから手動でデータをダウンロードできる。

### 方法3: ユーザが用意したraw dataから変換する

- すでに生データをダウンロードしている場合は、自分で原子ファイル形式に変換することもできる。
- すでに[RecDatasets](https://github.com/RUCAIBox/RecDatasets)でいくつかの変換スクリプトを公開している。(自作データセットを使う場合は、この変換スクリプトを参考にすれば良さそう。)

## DataFlow

- extensibility(拡張性) と reusability(再利用性)のために、data moduleは、**rawデータをモデルinputに変換する**エレガントなデータ・フローを設計している。
- Raw input:
  - 色んなデータを使える
- Atomic files:
  - raw inputをatomic filesに変換する必要がある。
- Datasetクラス:
  - 基本情報:
    - `torch.utils.data.Dataset`クラスを継承したクラス。
    - 生のデータセットをメモリ内に格納し、前処理の色んなfunctionを提供する。
  - method:
    - `build()`: datasetクラスを evalSettingsのconfigに従ってDataLoaderに変換する。
    - `save()`: `config['checkpoint_dir']`に指定された場所にDatasetオブジェクトを保存する。
    - `copy(new_inter_feat)`: 新しいinteraction featureを追加された、新しいDatasetオブジェクトを返す。
    - etc.
  - property:
    - dataset_name、dataset_path: データセットの名前とローカルファイルパス。
    - field2type、field2source: 特徴量名をその型やソースにマッピングする辞書。
    - uid_field、iid_field、label_field、time_field: ユーザーID、アイテムID、ラベル、タイムスタンプなどのフィールド名。
    - inter_feat、user_feat、item_feat: インタラクション、ユーザー、アイテムの特徴を格納する内部データ構造。内部データ構造はInteractionオブジェクト。
    - etc.
- DataLoaderクラス:
  - AbstractDataLoaderクラス:
    - 抽象クラス。
    - イテレーションごとにInteractionオブジェクトとしてロードされたデータのバッチを返す。
    - 他の全てのDataLoaderクラスは、このクラスを継承している。
    - 引数:
      - dataset: Datasetオブジェクト。
      - config: Configクラス。
      - sampler: Samplerクラス。(ネガティブサンプリング戦略クラス!)
      - shuffle: ラウンド後(=1 epoch後?)にデータをシャッフルするかどうかのbool値。(default: False)
  - general_dataloaders:
    - FullSortEvalDataLoaderクラス:
      - full-sort評価用のデータローダー。計算を高速化するために、ユーザーのインタラクション部分、ポジティブアイテム、usedアイテムのみを返し、ネガティブアイテムは返さない。
    - NegSampleEvalDataLoaderクラス:
      - ネガティブサンプリング評価用のデータローダー。
    - TrainDataLoaderクラス:
      - ネガティブサンプリング学習用のデータローダー。
      - config["training_neg_sample_num"]が0でない場合、ネガティブインタラクションを生成できる。
  - knowledge_aware_dataloaders:
    - KGDataLoaderクラス:
      - 知識グラフのnegative sampleを含むトリプレットを返すデータローダー。
    - KnowledgeBasedDataLoaderクラス:
      - 知識ベースモデルに使用される。
  - user_dataloaders:
    - UserDataLoaderクラス:
      - user_idを含むデータだけがbatchに含まれて返される。(??)
- Interactionクラス:
  - interactionレコード集合のbatchを表すクラス。
  - 属性:
    - interaction: dict or pd.DataFrame
  - 関数:
    - cat_interactions(list[Interaction]):複数のInteractionオブジェクトをconcatする。
  - Interactionクラス内のデータのルール
    - 学習時は特になし。
    - テスト時は、同一ユーザーのレコードが連続している必要がある。またその際は、positive exampleの後にnegative exampleが続く必要がある。

## Atomic Files

- RecBoleは**6つのアトミックファイルタイプ**を導入しており、suffixによって異なるファイルを識別している:
  - `.inter`: user-item interaction
  - `.user`: user feature
  - `.item`: item feature
  - `.kg`: triplets in a knowledge graph
  - `.link`: item-entity linkage data
  - `.net`: social graph data
- 各推薦タスク(i.e. 推薦手法)によって、必須ファイルが異なる:
  - Generalタスク: `.inter`が必須。
  - Context-awareタスク: `.inter`, `.user`, `item`が必須。
  - Knowledge-awareタスク: `.inter`, `.kg`, `.link`が必須。
  - Sequentialタスク: `.inter`が必須。
  - Socialタスク: `.inter`, `.net` が必須。
- Atomic fileのフォーマット:
  - 各ファイルは、$m \times n$ のテーブルで閲覧できる。
    - $m-1$: データレコード数。1行はheader
    - $n$: 特徴量の数。
  - header行の特徴量名は、`feat_name: feat_type`の形式で保持される。(特徴量名だけじゃないんだ...!:thinking:)
    - RecBoleは4種類の`feat_type`をサポートしてる:
      - `token`: single descrete feature (ex. user_id, age)
      - `token_seq`: discrete features sequence (ex. review text)
      - `float`: single continuous feature (ex. rating, timestamp)
      - `float_seq`: continuous features sequence (ex. vector)

### オリジナルで作ったAtomic fileを追加する方法

- オリジナルで作った特徴量(e.g. pretrained entity embeddings) を保存したAtomic fileを読み込ませたい場合:
- 第一に、オリジナルのatomic fileを用意する。(ex. `ml-1m.ent`)(extensionも新しくていいんだ...!:thinking:)
- 第二に、config (data config?) に引数を追加する:

```yaml
additional_feat_suffix: [ent] # 新しいsuffix(extension)を追加する
load_col:
  # inter/user/item/...:等の他のsuffixの設定と同様に...!
  ent: [ent_id, ent_emb]
```

- 第三に、`Dataset` objectとしてloadする。

```python
dataset = create_dataset(config)
print(dataset.ent_feat)
```

- これで他の特徴量と同様に前処理できるようになる。

## `Interaction` object

## Label of data

- explicit feedbackデータを扱う際に必要なやつ。
- データのラベルを3つの方法で設定できる。

### 方法1: `LABEL_FIELD`を設定する

- explicit feedbackの場合、configに`LABEL_FIELD`にラベルを表す特徴量名を指定し、且つ`train_neg_sample_args`にNoneを指定してnegative samplingを実行しないようにすればOK。

```yaml
LABEL_FIELD: label
train_neg_sample_args: None
```

- ラベルカラムの値は(0 or 1)のbinary値でなければならない点に注意!
  - (あれ? 5段階のratingとかををそのまま扱う事はできないのか...??:thinking:)

### 方法2: thresholdを設定する。

- binaryのラベルではなく、複数段階のratingなどのuser feedback情報を持っている場合、configにて閾値を設定してラベルを作ることができる。
- 例:

```yaml
threshold:
  rating: 3
train_neg_sample_args: None
```

### 方法3: Negative samplingを実行する(implicit feedbackの話?)

- implicit feedbackの場合、観測された全てのinteractionにpositive labelを設定し、**特定の戦略に基づいて**未観測のinteractionからnegative sampleをサンプリングする。
- configにて設定する。
- 例:

```yaml
train_neg_sample_args:
  uniform: 1
```

- ↑の例では、各positive sampleに対して、1つのnegativeサンプルを一様ランダムにサンプリングする。
- これらのconfig設定の詳細は、[Data settings](https://recbole.io/docs/user_guide/config/data_settings.html) と [Training Settings](https://recbole.io/docs/user_guide/config/training_settings.html) を参照。

# DataLoaderをカスタムするケース

- 必要なDataLoaderクラスを継承して、バッチ生成やネガティブサンプリング戦略に関わるメソッドを適宜オーバーライドする。
- 基本 `AbstractDataLoader` か `NegSampleDataLoader` のどちらかを継承する必要がありそう。軽微な変更で済むなら他のクラスを継承しても良いかも。
- AbstractDataLoaderクラスを継承する場合:
  - 3つの重要な属性:
    - pr: データローダーのポインター。
    - batch_size: バッチサイズ。
    - step: 各バッチのprの増分。
  - 実装すべき4つの関数:
    - `_init_batch_size_and_step`: バッチサイズとstepを初期化する。
    - `pr_end`: prの最大値+1を返す。
    - `_shuffle`: データセットの並び替え。
    - `_next_batch_data`: データローダークラスが`__next__()`で呼び出される際に、次のバッチのデータを返す。
- NegSampleDataLoaderクラスを継承する場合:
  - ネガティブサンプリングする場合は基本こちらを継承する。
  - 継承元のAbstractDataLoaderクラスの関数に加えて、以下の4つの関数を持つ。(必ずしも実装する必要はなく、あくまでNegSampleDataLoaderクラスの補助関数)
    - `_set_neg_sample_args`: ネガティブサンプリング戦略の設定を行う。
    - `_neg_sampling`: ネガティブサンプリングを実行する。
    - `_neg_sample_by_pair_wise_sampling`: ネガティブサンプリング戦略の一つ。pair-wise samplingを実行する。
    - `_neg_sample_by_point_wise_sampling`: ネガティブサンプリング戦略の一つ。point-wise samplingを実行する。
