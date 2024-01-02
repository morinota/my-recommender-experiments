import pandas as pd

# test_impressions_are_separated_as_implicit_feedbackが通るように修正してください


class ImpresionsSeparator:
    """MIND datasetのimpressionsカラムの中身を分割するクラス"""

    def __init__(self) -> None:
        pass

    def separate(
        self,
        behavior_df: pd.DataFrame,
        impressions_col: str = "impressions",
        separated_col: str = "news_id",
        label_col: str = "label",
    ) -> pd.DataFrame:
        """behavior_dfのimpressionsカラムの中身を分割し、behavior_dfからimpressions_colを削除しseparated_col & label_colを追加したpd.DataFrameを返す
        Args:
            behavior_df: MIND datasetのbehaviors.tsvをpd.read_tableしたもの
            impressions_col: behavior_dfのimpressionsカラムの名前
            separated_col: 分割されたimpressionsのうち、どのカラムを返すか
            label_col: 分割されたimpressionの各記事のclickしたか否かのbinary labelをどのカラムに追加するか
        Returns:
            分割されたimpressionsのうち、separated_colのみを返したpd.DataFrame
        """
        implicit_feedbacks = []
        remain_cols = behavior_df.drop(impressions_col, axis=1).columns.tolist()

        for _, record in behavior_df.iterrows():
            impressions_str = record[impressions_col]
            impressions_list = impressions_str.split()
            remain_value_by_col = {col: record[col] for col in remain_cols}
            print(remain_value_by_col)

            for impression in impressions_list:
                news_id, label = impression.split("-")
                implicit_feedbacks.append({**remain_value_by_col, separated_col: news_id, label_col: int(label)})

        return pd.DataFrame(implicit_feedbacks)
