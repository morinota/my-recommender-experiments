import unittest
from unittest.mock import MagicMock
import pandas as pd
from recommender_experiments.model.practice.split_dataset import Step50SplitDatasetTask


class TestStep50SplitDatasetTask(unittest.TestCase):
    def setUp(self):
        self.output_data = None

    def test_run(self):
        # Arrange
        sut = Step50SplitDatasetTask()
        sut.load = MagicMock(
            return_value=pd.DataFrame(
                {
                    "PUBLISHER": ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com"],
                    "CATEGORY": ["cat1", "cat2", "cat1", "cat3"],
                    "TITLE": ["title1", "title2", "title3", "title4"],
                }
            )
        )  # test doubleのstubとしての運用
        sut.dump = MagicMock(side_effect=self._dump)  # test doubleのmockとしての運用

        # Act
        sut.run()

        # Assert
        expected_output = (
            pd.DataFrame(
                {
                    "PUBLISHER": ["Reuters", "Huffington Post", "Businessweek"],
                    "CATEGORY": ["cat1", "cat2", "cat1"],
                    "TITLE": ["title1", "title2", "title3"],
                }
            ),
            pd.DataFrame({"PUBLISHER": ["Contactmusic.com"], "CATEGORY": ["cat3"], "TITLE": ["title4"]}),
            pd.DataFrame({"PUBLISHER": ["Daily Mail"], "CATEGORY": ["cat2"], "TITLE": ["title5"]}),
        )
        # pd.testing.assert_frame_equal(self.output_data[0], expected_output[0])
        # pd.testing.assert_frame_equal(self.output_data[1], expected_output[1])
        # pd.testing.assert_frame_equal(self.output_data[2], expected_output[2])
        assert type(self.output_data[0]) == pd.DataFrame
        assert type(self.output_data[1]) == pd.DataFrame
        assert type(self.output_data[2]) == pd.DataFrame

    def _dump(self, data):
        self.output_data = data
