from feature_pipelines.impression_separator import ImpresionsSeparator
import pandas as pd


def test_impressions_are_separated_as_implicit_feedback() -> None:
    behavior_df = pd.DataFrame(
        {
            "impression_id": [0],
            "user_id": [1],
            # "MM/DD/YYYY HH:MM:SS AM/PM"という形式
            "time": ["01/01/2024 9:15:32 AM"],
            "history": ["N0 N6"],
            "impressions": ["N1-1 N2-1 N3-0 N4-0 N5-0"],
        }
    )
    impressions_col = "impressions"
    separated_col = "news_id"
    sut = ImpresionsSeparator()

    implicit_feedbacks = sut.separate(behavior_df, impressions_col, separated_col)

    implicit_feedbacks_expected = behavior_df = pd.DataFrame(
        {
            "impression_id": [0, 0, 0, 0, 0],
            "user_id": [1, 1, 1, 1, 1],
            "news_id": ["N1", "N2", "N3", "N4", "N5"],
            "time": [
                "01/01/2024 9:15:32 AM",
                "01/01/2024 9:15:32 AM",
                "01/01/2024 9:15:32 AM",
                "01/01/2024 9:15:32 AM",
                "01/01/2024 9:15:32 AM",
            ],
            "label": [1, 1, 0, 0, 0],
            "history": ["N0 N6", "N0 N6", "N0 N6", "N0 N6", "N0 N6"],
        }
    )
    assert type(implicit_feedbacks) == pd.DataFrame
    assert len(implicit_feedbacks) == len(implicit_feedbacks_expected)
    assert implicit_feedbacks.to_dict() == implicit_feedbacks_expected.to_dict()
