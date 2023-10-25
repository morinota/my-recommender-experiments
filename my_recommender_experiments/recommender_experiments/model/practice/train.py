import gokart
from sklearn.linear_model import LogisticRegression
from extract_feature import Step51ExtractFeatureTask

from split_dataset import Step50SplitDatasetTask


class Step52TrainTask(gokart.TaskOnKart):
    def requires(self) -> dict[str, gokart.TaskOnKart]:
        return {"data": Step50SplitDatasetTask(), "feature": Step51ExtractFeatureTask()}
        # 依存先の上流タスクが複数ある場合、このようにdictで登録する。

    def run(self):
        df_train, _, _ = self.load("data")  # loadもdictのkeyを指定して上流タスクのdumpを取得する。
        X_train, _, _ = self.load("feature")
        y_train = df_train["CATEGORY"]

        model = LogisticRegression(random_state=123, max_iter=10000)
        model.fit(X_train, y_train)

        self.dump(model)  # 学習したモデルはいつも通り dump する
