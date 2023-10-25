from io import BytesIO
import gokart
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from predict import Step53PredictTask

from split_dataset import Step50SplitDatasetTask


class Step55CalcConfusionMatrixTask(gokart.TaskOnKart):
    def output(self):
        return self.make_target("test_confusion_matrix.png")

    def requires(self):
        # 同様に複数の依存先タスクを登録
        return {"data": Step50SplitDatasetTask(), "pred": Step53PredictTask()}

    def run(self):
        _, _, df_test = self.load("data")
        pred_test = self.load("pred")["pred"]
        gt_test = df_test["CATEGORY"]
        test_cm = confusion_matrix(gt_test, pred_test)

        fig = plt.figure()
        # sns.heatmap(test_cm, annot=True, cmap="Blues")
        figbin = BytesIO()
        fig.savefig(figbin, format="png")
        self.dump(figbin.getvalue())
