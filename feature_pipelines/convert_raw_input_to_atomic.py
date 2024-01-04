import pandas as pd

from task_interface import TaskInterface


class ConvertRawInputToAtomicTask(TaskInterface):
    SEPARATOR = "\t"
    LINEBREAK = "\n"

    def __init__(self) -> None:
        pass

    def run(
        self,
        df: pd.DataFrame,
        feature_type_by_name: dict[str, str],
    ) -> str:
        output_data = pd.DataFrame()
        print(f"feature_type_by_name: {feature_type_by_name=}")
        print(f"df.columns: {df.columns=}")
        for feature_name, feature_type in feature_type_by_name.items():
            atomic_feature_name = f"{feature_name}:{feature_type}"
            output_data[atomic_feature_name] = df.loc[:, feature_name]

        # 1行目はfield名
        atomic_records = [self.SEPARATOR.join(output_data.columns)]

        for record in output_data.itertuples():
            atomic_record = self.SEPARATOR.join([str(feature) for feature in record[1:]])  # idx=0は行番号なので除外
            atomic_records.append(atomic_record)

        return self.LINEBREAK.join(atomic_records)
