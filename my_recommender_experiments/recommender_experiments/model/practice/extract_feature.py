from split_dataset import Step50SplitDatasetTask
import gokart
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class Step51ExtractFeatureTask(gokart.TaskOnKart):
    def requires(self):
        return Step50SplitDatasetTask()  # 依存先の上流タスクを定義する。

    def run(self) -> None:
        df_train, df_valid, df_test = self.load()  # 依存先タスクがdumpしたデータを取得する。

        # 10回以上出現する unigram, bi-gram について計算
        vec_tfidf = TfidfVectorizer(min_df=10, ngram_range=(1, 2))

        # valid は tf-idf を計算するための train data に含めても良いが、今回はやらない
        tfidf_train = vec_tfidf.fit_transform(df_train["TITLE"])
        tfidf_valid = vec_tfidf.transform(df_valid["TITLE"])
        tfidf_test = vec_tfidf.transform(df_test["TITLE"])

        # DataFrame に変換
        df_train = pd.DataFrame(tfidf_train.toarray(), columns=vec_tfidf.get_feature_names_out())
        df_valid = pd.DataFrame(tfidf_valid.toarray(), columns=vec_tfidf.get_feature_names_out())
        df_test = pd.DataFrame(tfidf_test.toarray(), columns=vec_tfidf.get_feature_names_out())

        # 今回は使用しないが一応保存
        df_train.to_csv("output/train.feature.txt", index=None, sep="\t")
        df_valid.to_csv("output/valid.feature.txt", index=None, sep="\t")
        df_test.to_csv("output/test.feature.txt", index=None, sep="\t")

        self.dump((df_train, df_valid, df_test))  # 次のタスクにわたす為にdump
