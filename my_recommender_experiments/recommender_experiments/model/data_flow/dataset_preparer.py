from pathlib import Path

from recommender_experiments.dataset.MIND_dataset import MINDDataset
from recommender_experiments.model.data_flow.convert_raw_input_to_atomic import ConvertRawInputToAtomicTask
from recommender_experiments.model.data_flow.raw_input_downloader import DownloadRawInputTask
from recommender_experiments.model.data_flow.atomic_file_exporter import AtomicFileExporter

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
            "behavior.inter": converter.run(mind_dataset.behaviors, mind_dataset.behaviors_fields),
            "news.item": converter.run(mind_dataset.news, mind_dataset.news_fields),
            "entity_embeddings.ent": converter.run(
                mind_dataset.entity_embeddings,
                mind_dataset.entity_embedding_fields,
            ),
            "relation_embeddings.rel": converter.run(
                mind_dataset.relation_embeddings,
                mind_dataset.relation_embedding_fields,
            ),
        }
        print("[LOG] convert_to_atomic finished")

        exporter = AtomicFileExporter()
        return exporter.run(atomic_data_by_name, destination_dir)
