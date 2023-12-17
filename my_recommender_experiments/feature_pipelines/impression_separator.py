import pandas as pd

# test_impressions_are_separated_as_implicit_feedbackが通るように修正してください


class ImpresionsSeparator:
    """MIND datasetのimpressionsカラムの中身を分割するクラス"""

    def __init__(self) -> None:
        pass

    def separate(
        self,
        behavior_df: pd.DataFrame,
        impressions_col_name: str = "impressions",
        separated_col_name: str = "news_id",
    ) -> pd.DataFrame:
        """behavior_dfのimpressionsカラムの中身を分割し、behavior_dfからimpressions_colを削除しseparated_colを追加したpd.DataFrameを返す
        Args:
            behavior_df: MIND datasetのbehaviors.tsvをpd.read_tableしたもの
            impressions_col: behavior_dfのimpressionsカラムの名前
            separated_col: 分割されたimpressionsのうち、どのカラムを返すか
        Returns:
            implicit_feedbacks:
        """
        implicit_feedbacks = []
        remain_cols = behavior_df.drop(impressions_col_name, axis=1).columns.tolist()

        for _, record in behavior_df.iterrows():
            impressions_str = record[impressions_col_name]
            impressions_list = impressions_str.split()
            remain_value_by_col = {col: record[col] for col in remain_cols}

            for impression in impressions_list:
                news_id, is_interact = impression.split("-")
                if is_interact == "0":
                    continue
                implicit_feedbacks.append({**remain_value_by_col, separated_col_name: news_id})

        return pd.DataFrame(implicit_feedbacks)
