# MIND: MIcrosft News Dataset

- [MIND: MIcrosft News Dataset](https://msnews.github.io/)

- MIND-smallデータセット:
  - 研究者がMINDのデータに慣れ、迅速な実験を行えるように、MINDデータセットから5万人のユーザとその行動ログを無作為に抽出(=分布は同じ?)し、MINDデータセットの小型版を公開
  - MIND-smallのトレーニングセットと検証セットは、[以下](https://msnews.github.io/#:~:text=The%20training%20and%20validation%20sets%20of%20MIND%2Dsmall%20can%20be%20downloaded%20at%3A)からダウンロードできる。
- いくつかの既存のニュース推薦手法と一般的な推薦手法の実装は、[Microsoft Recommenders](https://github.com/microsoft/recommenders)で見ることができる。
- データセットの予測を生成するためのステップ・バイ・ステップのチュートリアルは[以下](https://msnews.github.io/#:~:text=Following%20is%20a%20step%2Dby%2Dstep%20tutorial%20for%20generating%20predictions%20on%20the%20dataset%3A)。

## データセットの中身:

- 参考: https://learn.microsoft.com/ja-jp/azure/open-datasets/dataset-microsoft-news?tabs=azureml-opendatasets
  - 4種類のデータ(train_small, valid_small, train_large, valid_large)の仕様は同じ。
- behaviors.tsv:
  - ユーザーのクリック履歴とインプレッション ログ
  - columns:
    - impression_id:
    - user_id:
    - time: "MM/DD/YYYY HH:MM:SS AM/PM"という形式。
    - history: このユーザーの、このインプレッションより前のニュース クリック履歴 (クリックしたニュースの ID リスト)(丁寧なデータセット...!)
    - impressions: ?
- news.tsv:
  - ニュース記事の属性情報。
  - columns:
    - news_id, category, subcategory, title, summary, url, title_entities, summary_entities,
- entity_embedding.vec:
  - knowledge graph から抽出したニュースに含まれるエンティティ(=固有名詞??)の埋め込み。
  - ?
- relation_embedding.vec:
  - knowledge graph から抽出したエンティティ間の関係の埋め込み。
  - ?
