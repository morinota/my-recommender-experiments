from pathlib import Path
from atomic_coverter.mind_converter import MINDTrainConverter


class AtomicFileFetcher:
    CONVERTER_BY_DATASET_NAME = {
        "mind_training_small": MINDTrainConverter,
        "mind_train_large": MINDTrainConverter,
    }

    def __init__(self):
        pass

    def fetch(self, raw_data_dir: Path, atomic_file_dir: Path, dataset_name: str) -> None:
        converter_class = self.CONVERTER_BY_DATASET_NAME[dataset_name]
        converter = converter_class(raw_data_dir, atomic_file_dir, dataset_name)
        converter.convert()
