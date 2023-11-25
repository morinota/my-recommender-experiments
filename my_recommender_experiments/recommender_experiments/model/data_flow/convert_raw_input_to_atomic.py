import os
from pathlib import Path
import tempfile

import zipfile
import gokart


from recommender_experiments.model.task_interface import TaskInterface
from recommender_experiments.model.data_flow.download_raw_input import DownloadRawInputTask
from recommender_experiments.dataset.MIND_dataset import MINDDataset


class ConvertRawInputToAtomicTask(TaskInterface):
    zip_path = gokart.TaskInstanceParameter()

    def __init__(self, raw_input_zip_path: Path) -> None:
        self.raw_input_zip_path = raw_input_zip_path

    def run(self) -> tuple[Path, Path, Path, Path]:
        zip_path: Path = self.load("zip_path")
        output_path = Path(tempfile.gettempdir()) / "mind"
        output_path.mkdir(parents=True, exist_ok=True)
        dataset_kind = "validation_small"
        return self._run(zip_path, output_path, dataset_kind)

    @staticmethod
    def _run(
        zip_path: Path,
        output_path: Path,
        dataset_kind: str,
    ) -> tuple[Path, Path, Path, Path]:
        temp_dir = Path(tempfile.gettempdir()) / "mind"
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        dataset = MINDDataset(temp_dir, output_path, dataset_kind)

        behaviors_path = dataset.convert_behaviors()
        news_path = dataset.convert_news()
        entity_embedding_path = dataset.convert_entity_embedding()
        relation_embedding_path = dataset.convert_relation_embedding()
        return (
            behaviors_path,
            news_path,
            entity_embedding_path,
            relation_embedding_path,
        )
