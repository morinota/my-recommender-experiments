import pandas as pd

from recommender_experiments.model.task_interface import TaskInterface


class ConvertRawInputToAtomicTask(TaskInterface):
    SEPARATOR = "\t"
    LINEBREAK = "\n"

    def __init__(self) -> None:
        pass

    def run(
        self,
        df: pd.DataFrame,
        field_type_by_id: dict[int, str],
    ) -> str:
        output_data = pd.DataFrame()
        print(f"field_type_by_id: {field_type_by_id=}")
        print(f"df.columns: {df.columns=}")
        for col_idx in field_type_by_id:
            output_data[col_idx] = df.iloc[:, col_idx]
        print(f"hoge")

        # 1行目はfield名
        atomic_records = [self.SEPARATOR.join([field_type_by_id[int(col_idx)] for col_idx in output_data.columns])]

        for record in output_data.itertuples():
            atomic_records.append(
                self.SEPARATOR.join([str(record[int(col_idx) + 1]) for col_idx in output_data.columns])
            )

        return self.LINEBREAK.join(atomic_records)
