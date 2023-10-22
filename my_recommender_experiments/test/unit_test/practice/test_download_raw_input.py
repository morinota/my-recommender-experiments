import unittest
from unittest.mock import MagicMock
import pandas as pd
from recommender_experiments.model.data_flow.download_raw_input import DownloadRawInputTask


class TestDownloadRawInputTask(unittest.TestCase):
    def setUp(self):
        self.output_data = None

    def test_run(self):
        # Arrange
        sut = DownloadRawInputTask("validation_small")
        sut.dump = MagicMock(side_effect=self._dump)  # test doubleのmockとしての運用

        # Act
        sut.run()

        # Assert

    def _dump(self, data):
        self.output_data = data
