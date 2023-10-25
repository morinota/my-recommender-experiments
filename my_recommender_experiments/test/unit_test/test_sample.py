from logging import getLogger
import unittest
from unittest.mock import MagicMock

from recommender_experiments.model.sample import Sample

logger = getLogger(__name__)


class TestSample(unittest.TestCase):
    def setup(self):
        self.output_data = None

    def test_text_sample_output_is_dumped(self):
        # Arrange
        sut = Sample()
        sut.dump = MagicMock(side_effect=self._dump)  # taskの出力をself.outputに渡す指定(テストダブルのmockとしての運用か...!)

        # Act
        sut.run()

        # Assert
        # self.assertEqual(self.output_data, "sample output")
        assert self.output_data == "sample output"  # どっちでもOK

    def _dump(self, data):
        self.output_data = data
