import os
from pathlib import Path
import tempfile
from typing import Optional
import zipfile
import gokart
import luigi
from recommender_experiments.dataset.data_config import MINDConfig
from urllib import request


class DownloadRawInputTask(gokart.TaskOnKart):
    dadaset_name_candidate = [
        "training_small",
        "validation_small",
        "training_large",
        "validation_large",
    ]
    dataset_name = luigi.TaskParameter(
        "training_small",
        description="",
    )

    def run(self) -> None:
        temp_dir = Path(tempfile.gettempdir()) / "mind"
        temp_dir.mkdir(parents=True, exist_ok=True)

        self.dump(
            self._run(
                url=MINDConfig.validation_small_url,
                destination_filepath=temp_dir / "validation_small.tsv",
            )
        )

    @staticmethod
    def _run(
        url: str,
        destination_filepath: Path,
        is_force_download: bool = False,
    ) -> Path:
        if (not is_force_download) and (destination_filepath.exists()):
            print(f"Bypassing download of already-downloaded file: {str(destination_filepath)}")
            return destination_filepath

        print(f"Downloading file {os.path.basename(url)} to {str(destination_filepath)}")

        with request.urlopen(url) as response, destination_filepath.open("wb") as out_file:
            data = response.read()
            out_file.write(data)
        assert destination_filepath.is_file()
        nBytes = destination_filepath.stat().st_size
        print(f"...done, {nBytes} bytes.")

        return destination_filepath


if __name__ == "__main__":
    task = DownloadRawInputTask(dataset_name="validation_small")
    task.run()
