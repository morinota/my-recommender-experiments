このリポジトリは、推薦システムに関する実験や研究のためのプロジェクトです。さまざまな推薦アルゴリズムやデータセットを利用し、効果的な推薦システムの開発や評価を行います。また、実験結果や比較データなどもこのリポジトリで管理されます。

# 開発の方針:

- 参考: https://qiita.com/wakame1367/items/a41708c970932c2c724f#git%E7%AD%89%E3%81%A7%E7%AE%A1%E7%90%86%E3%81%97%E3%81%A6%E3%81%84%E3%82%8B%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E6%A7%8B%E9%80%A0%E3%82%92%E3%81%9D%E3%81%AE%E3%81%BE%E3%81%BE%E4%BD%BF%E3%81%86
- easy_goldディレクトリ以下にコードを書く。
- kaggleのcode competitionをする上では、最終的にsubmission.csvが出力されるようにコードを書く。
  - build.pyを実行する。
    - やっている事は以下:
    - 1. 全てのコードをbase64でencodeして文字列に変換する。
    - 2. encodeしたコードのファイルパスをkey, encodeした文字列をvalueにしたdictを作る。
    - 3. script_template.pyというscript.pyを作るためのテンプレートの`file_data`部分に辞書の内容を置換する。
  - 実行後、build/script.pyが生成されているので、それをNotebooksに貼り付ける。
  - その後はCommitボタンを押してsubmitするという流れ。

## 実行環境link

https://www.kaggle.com/masatomasamasa/my-recommender-experiments/edit

windowsの場合は、numpyとscipyをhttps://toriaezu-engineer.hatenablog.com/entry/2016/10/09/084428を参考にinstallする。
