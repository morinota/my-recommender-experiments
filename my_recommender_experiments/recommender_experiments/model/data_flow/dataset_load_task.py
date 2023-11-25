from pathlib import Path
from typing import Any
from my_recommender_experiments.recommender_experiments.model.data_flow.raw_input_reader import ReadRawInputTask
from recommender_experiments.model.data_flow.convert_raw_input_to_atomic import ConvertRawInputToAtomicTask
from recommender_experiments.model.data_flow.download_raw_input import DownloadRawInputTask
from recommender_experiments.model.task_interface import TaskInterface


class DatasetLoaderTask(TaskInterface):
    def __init__(self) -> None:
        pass

    def run(self, dataset_type: str, destination_dir: Path) -> Any:
        downloader = DownloadRawInputTask(dataset_type, destination_dir)
        raw_input_zip_path = downloader.run()

        raw_input_reader = ReadRawInputTask()
        raw_inputs = raw_input_reader.run(raw_input_zip_path)

        converter = ConvertRawInputToAtomicTask()
        atomic_dataset = converter.run(raw_inputs)
