from logging import getLogger

from recommender_experiments.utils.template import GokartTask
import gokart

logger = getLogger(__name__)


class Sample(GokartTask):
    def run(self):
        self.dump("sample output")  # 出力


class StringToSplit(GokartTask):
    """Like the function to divide received data by spaces."""

    task = gokart.TaskInstanceParameter(expected_type=Sample)  # こういうタスクのつなぎ方もできるんだ。

    def run(self):
        sample = self.load("task")
        self.dump(sample.split(" "))


class Main(GokartTask):
    """Endpoint task."""

    def requires(self) -> gokart.TaskOnKart:
        # 依存先のタスクを登録
        return StringToSplit(task=Sample())
