from pathlib import Path

from MIND_dataset import MINDDataset
from convert_raw_input_to_atomic import ConvertRawInputToAtomicTask
from raw_input_downloader import DownloadRawInputTask
from atomic_file_exporter import AtomicFileExporter
from task_interface import TaskInterface
import zipfile


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

        # zipファイルを同じ場所にunzipする
        unzip_dir = raw_input_zip_path.parent
        with zipfile.ZipFile(raw_input_zip_path, "r") as zip_ref:
            zip_ref.extractall(unzip_dir)
        print("[LOG] unzip finished")
        # TODO:特徴量エンジニアリングすることを考えると、一旦読み込んでから、特徴量を作って、カラム付きでtsv出力し直すと良さそう。

        # mind_dataset = MINDDataset.load_from_zip(raw_input_zip_path)
        # print("[LOG] MINDDataset.load_from_zip finished")

        # mind_dataset.export(output_dir=destination_dir)
        # print("[LOG] mind_dataset.export finished")
