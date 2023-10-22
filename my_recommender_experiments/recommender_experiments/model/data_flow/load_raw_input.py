import os
from pathlib import Path
import tempfile
from typing import Optional
import zipfile
import gokart
import luigi
import pandas as pd
from recommender_experiments.dataset.data_config import MINDConfig
from recommender_experiments.model.data_flow.download_raw_input import DownloadRawInputTask

from urllib import request


class LoadRawInputTask(gokart.TaskOnKart):
    dadaset_name_candidate = [
        "training_small",
        "validation_small",
        "training_large",
        "validation_large",
    ]

    def requires(self):
        return DownloadRawInputTask()

    def run(self) -> None:
        zip_path: Path = self.load()
        self.dump(self._run(zip_path))

    @staticmethod
    def _run(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        temp_dir = Path(tempfile.gettempdir()) / "mind"
        temp_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        behaviors_path = temp_dir / "behaviors.tsv"
        behaviors_df = pd.read_table(
            behaviors_path,
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )

        news_path = temp_dir / "news.tsv"
        news_df = pd.read_table(
            news_path,
            header=None,
            names=["id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"],
        )

        return behaviors_df, news_df


if __name__ == "__main__":
    task = LoadRawInputTask(dataset_name="validation_small")
    task.run()
