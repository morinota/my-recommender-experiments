from sklearn.model_selection import train_test_split
from recommender_experiments.model.practice.download_dataset import Step50DownloadDatasetTask
import gokart
import pandas as pd


class Step50SplitDatasetTask(gokart.TaskOnKart):
    def requires(self):
        return Step50DownloadDatasetTask()

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df: pd.DataFrame = self.load()  # これで依存先のtaskでdumpしたものが読み込まれる。
        # このようにしてパイプラインを作っていく。
        # 依存関係はgokartが面倒見てくれるので、上流タスクを再実行した場合は下流タスクも再実行されるようになる。

        # 50-2: 情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する．
        pub_list = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]
        df = df[df["PUBLISHER"].isin(pub_list)]

        seed = 12345

        # 50-3, 50-4: scikit-learn の train_test_split では2つにしか分割できないため2回に分けて3つに分割する (https://datascience.stackexchange.com/a/15136/126697)
        df_train, df_valid_test = train_test_split(df, test_size=0.5, random_state=seed)
        df_valid, df_test = train_test_split(df_valid_test, test_size=0.5, random_state=seed)

        # 50-4: それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する
        # 今回は使わないが一応保存しておく
        # df_train[["CATEGORY", "TITLE"]].to_csv("output/train.txt", header=None, index=None, sep="\t")
        # df_valid[["CATEGORY", "TITLE"]].to_csv("output/valid.txt", header=None, index=None, sep="\t")
        # df_test[["CATEGORY", "TITLE"]].to_csv("output/test.txt", header=None, index=None, sep="\t")

        # 学習データと評価データを作成したら，各カテゴリの事例数を確認せよ．
        print("Train Data")
        print(df_train["CATEGORY"].value_counts())
        print("Validation Data")
        print(df_valid["CATEGORY"].value_counts())
        print("Test Data")
        print(df_test["CATEGORY"].value_counts())

        # データを分割した後、それぞれdumpして、次のタスクでつかえるようにする。
        self.dump((df_train, df_valid, df_test))
