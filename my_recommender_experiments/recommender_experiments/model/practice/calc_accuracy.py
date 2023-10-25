import gokart

# from sklearn.base import accuracy_score
from predict import Step53PredictTask

from split_dataset import Step50SplitDatasetTask


class Step54CalcAccuracyTask(gokart.TaskOnKart):
    def output(self):
        # デフォルトではpickleで保存されるが、今回は確認しやすいようにテキストファイルで保存したい。
        # この場合は、保存先に.txtを指定する。
        return self.make_target("test_accuracy.txt")

    def requires(self):
        return {"data": Step50SplitDatasetTask(), "pred": Step53PredictTask()}

    def run(self):
        _, _, df_test = self.load("data")
        pred_test = self.load("pred")["pred"]
        gt_test = df_test["CATEGORY"]

        test_accuracy = ""
        # ccuracy_score(gt_test, pred_test)
        self.dump(test_accuracy)
