import gokart
from sklearn.linear_model import LogisticRegression
from extract_feature import Step51ExtractFeatureTask

from split_dataset import Step50SplitDatasetTask
from train import Step52TrainTask


class Step53PredictTask(gokart.TaskOnKart):
    """
    学習したモデルで記事タイトルからカテゴリとその予測確率を計算する。
    52と同様に複数指定してロードする。
    また、保存 (dump) も辞書型でやってみる。 (https://gokart.readthedocs.io/en/latest/task_on_kart.html#taskonkart-dump)
    """

    def requires(self):
        # 依存先の上流タスクをdictで複数定義する。
        return {
            "data": Step50SplitDatasetTask(),
            "feature": Step51ExtractFeatureTask(),
            "model": Step52TrainTask(),
        }

    def output(self):
        # dumpの形式を定義してる。それぞれのkeyと保存先を定義。
        return {"pred": self.make_target("pred.pkl"), "prob": self.make_target("prob.pkl")}

    def run(self):
        _, _, df_test = self.load("data")
        _, _, X_test = self.load("feature")
        model: LogisticRegression = self.load("model")
        y_test = df_test["CATEGORY"]

        pred_test = model.predict(X_test)
        prob_test = model.predict_proba(X_test)

        self.dump(pred_test, "pred")  # valueオブジェクト, keyの順で渡す。outputで指定した保存先にdumpされる。
        self.dump(prob_test, "prob")
