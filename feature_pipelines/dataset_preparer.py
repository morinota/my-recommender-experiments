from pathlib import Path

from feature_pipelines.MIND_dataset import MINDDataset
from feature_pipelines.convert_raw_input_to_atomic import ConvertRawInputToAtomicTask
from feature_pipelines.raw_input_downloader import DownloadRawInputTask
from feature_pipelines.atomic_file_exporter import AtomicFileExporter

from recommender_experiments.model.task_interface import TaskInterface


class DatasetPreparer(TaskInterface):
    def __init__(self) -> None:
        pass

    def run(
        self,
        dataset_type: str,
        destination_dir: Path,
        is_force_download: bool = False,
    ) -> list[Path]:
        downloader = DownloadRawInputTask()
        raw_input_zip_path = downloader.run(dataset_type, destination_dir, is_force_download)
        print("[LOG] downloader.run finished")

        mind_dataset = MINDDataset.load_from_zip(raw_input_zip_path)
        print("[LOG] MINDDataset.load_from_zip finished")

        converter = ConvertRawInputToAtomicTask()
        atomic_data_by_name = {
            f"{dataset_type}.inter": converter.run(mind_dataset.behaviors, mind_dataset.behaviors_fields),
            f"{dataset_type}.item": converter.run(mind_dataset.news, mind_dataset.news_fields),
        }
        print("[LOG] convert_to_atomic finished")

        exporter = AtomicFileExporter()
        return exporter.run(atomic_data_by_name, destination_dir)