# MIND: MIcrosft News Dataset

## 参考:

- [MIND: MIcrosft News Dataset](https://msnews.github.io/)
- [Introduction to MIND and MIND-small datasets](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md)

## 概要

- ニュース推薦のためのデータセット:
  - Microsoft Newsウェブサイトの匿名化された行動ログから収集された。
  - 2019年10月12日から11月22日までの6週間の間に少なくとも5回のニュースクリックがあった100万人のユーザをランダムにサンプリングしたもの。
  - **6週目のログをtestデータ**(=leader boardのやつ)に使用し、**5週目のログをtrain** (=正確にはtrainデータ + validデータ)に使用した。
    - trainデータのサンプルについては、**1~4週目のクリック行動**を用いて、ユーザモデリングのためのニュースクリック履歴(=historyカラム)を構築した。
    - 学習データ(=train+valid)のうち、**5週目の最終日のサンプル**をvalidセットとして使用した。
  - また、50,000人のユーザとその行動ログを無作為に抽出し、MINDの小型版（MIND-small）をリリースした。MIND-smallデータセットには、trainデータとvalidデータのみが含まれている。(MINDの方はtestデータも含まれてるっぽい)

### MINDとMIND-small

- MIND-smallデータセット:
  - 研究者がMINDのデータに慣れ、迅速な実験を行えるように、MINDデータセットから5万人のユーザとその行動ログを無作為に抽出(=分布は同じ?)し、MINDデータセットの小型版を公開
  - MIND-smallのtrainセットとvalidセットは、[以下](https://msnews.github.io/#:~:text=The%20training%20and%20validation%20sets%20of%20MIND%2Dsmall%20can%20be%20downloaded%20at%3A)からダウンロードできる。
- いくつかの既存のニュース推薦手法と一般的な推薦手法の実装は、[Microsoft Recommenders](https://github.com/microsoft/recommenders)で見ることができる。
- データセットの予測を生成するためのステップ・バイ・ステップのチュートリアルは[以下](https://msnews.github.io/#:~:text=Following%20is%20a%20step%2Dby%2Dstep%20tutorial%20for%20generating%20predictions%20on%20the%20dataset%3A)。

## データセットの中身(4つのファイル)

- 参考: https://learn.microsoft.com/ja-jp/azure/open-datasets/dataset-microsoft-news?tabs=azureml-opendatasets
  - 4種類のデータ(train_small, valid_small, train_large, valid_large)の仕様は同じ。

## ユーザの行動履歴

- `behaviors.tsv`:
  - ユーザのクリック履歴。各recordは1回のimpressionを意味する。
  - columns:
    - impression_id:
    - user_id:
    - time:
      - "MM/DD/YYYY HH:MM:SS AM/PM"という形式。
    - history:
      - 対象ユーザの、対象impression timeより前の閲覧履歴(クリックしたニュースIDのlist)
      - trainデータの場合は、1~4週目の間の閲覧履歴。
      - timeの順(=たぶん昇順)にソートされている。
    - impressions:
      - 一回のimpression内で表示したニュース一覧。(5週目の間に読んだ記事ってこと??)
      - formatは`[News ID 1]-[label1] ... [News ID n]-[labeln]`
      - ex. `N4-1 N34-1 N156-0 N207-0 N198-0`
      - labelは対象ユーザによってclickされたか否か。(1 or 0)
      - 順番はランダムになってる。
  - user click historyとimpression newsの全ての記事は、news data fileに含まれている。

### 記事のメタデータ

- `news.tsv`:
  - ニュース記事の属性情報(メタデータ)。
  - columns:
    - news_id:
      - ex. `N37378`
    - category:
      - ex. `sports`
    - subcategory:
      - ex. `golf`
    - title:
      - `PGA Tour winners`
    - summary:
      - `A gallery of recent winners on the PGA Tour.`
    - url
    - title_entities:
      - タイトル内の固有名詞
      - ex.`[{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": [0], "SurfaceForms": ["PGA Tour"]}]`各key-valueの意味:
    - summary_entities:
      - summary内の固有名詞
      - ex. `[{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": [35], "SurfaceForms": ["PGA Tour"]}]`

entityの各key-valueの意味:

- Label: The entity name in the Wikidata knwoledge graph
- Type: The type of this entity in Wikidata
- WikidataId: The entity ID in Wikidata
- Confidence: The confidence of entity linking
- OccurrenceOffsets: The character-level entity offset in the text of title or abstract
- SurfaceForms: The raw entity names in the original text

### entityの埋め込み

- `entity_embedding.vec`:
  - knowledge graph から抽出したニュースに含まれるエンティティ(=固有名詞??)の埋め込み。
  - ?
- `relation_embedding.vec`:
  - knowledge graph から抽出したエンティティ間の関係の埋め込み。
  - ?
