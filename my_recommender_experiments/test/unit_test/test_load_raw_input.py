from pathlib import Path
import unittest
from unittest.mock import MagicMock
import pandas as pd
from recommender_experiments.model.data_flow.load_raw_input import LoadRawInputTask


class TestDownloadRawInputTask(unittest.TestCase):
    def test_run(self):
        # Arrange
        zip_filename = r"C:\Users\Masat\src\my-recommender-experiments\my_recommender_experiments\MINDsmall_train.zip"
        sut = LoadRawInputTask()

        # Act
        behavior_df, news_df = sut._run(Path(zip_filename))

        # Assert
        assert type(behavior_df) == pd.DataFrame
        assert type(news_df) == pd.DataFrame
