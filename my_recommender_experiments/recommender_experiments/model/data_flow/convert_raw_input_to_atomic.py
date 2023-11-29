import pandas as pd

from recommender_experiments.model.task_interface import TaskInterface


class ConvertRawInputToAtomicTask(TaskInterface):
    SEPARATOR = "\t"

    def __init__(self) -> None:
        pass

    def run(
        self,
        df: pd.DataFrame,
        field_type_by_id: dict[int, str],
    ) -> str:
        output_data = pd.DataFrame()
        for col_idx in field_type_by_id:
            output_data[col_idx] = df.iloc[:, col_idx]

        # 1行目はfield名
        atomic_str = self.SEPARATOR.join([field_type_by_id[int(col_idx)] for col_idx in output_data.columns]) + "\n"
        for record in output_data.itertuples():
            atomic_str += self.SEPARATOR.join([str(record[int(col_idx) + 1]) for col_idx in output_data.columns]) + "\n"
        return atomic_str
