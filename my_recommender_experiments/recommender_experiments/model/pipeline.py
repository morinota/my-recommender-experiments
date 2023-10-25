from pathlib import Path
from typing import Any

import luigi
from recommender_experiments.model.data_flow import DownloadRawInputTask, ConvertRawInputToAtomicTask

import gokart


class PipelineTask(gokart.TaskOnKart):
    destination_dir: Path = luigi.Parameter()

    def requires(self) -> dict[str, Any]:
        dataset_path = DownloadRawInputTask(destination_dir=self.destination_dir)

        atomic_file_pathes = ConvertRawInputToAtomicTask(zip_path=dataset_path)
        return atomic_file_pathes

    def run(self) -> None:
        self.dump("done")
