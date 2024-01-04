import os
from pathlib import Path
from task_interface import TaskInterface
from urllib import request


class DownloadRawInputTask(TaskInterface):
    DOWNLOAD_BASE_URL = "https://mind201910small.blob.core.windows.net/release"
    DATASET_FILE_BY_TYPE = {
        "mind_training_small": "MINDsmall_train.zip",  # about 50MB
        "mind_validation_small": "MINDsmall_dev.zip",  # about 30MB
        "mind_training_large": "MINDlarge_train.zip",
        "mind_validation_large": "MINDlarge_dev.zip",
    }

    def __init__(self) -> None:
        pass

    def run(
        self,
        dataset_type: str,
        destination_dir: Path,
        is_force_download: bool = False,
    ) -> Path:
        dataset_filename = self.DATASET_FILE_BY_TYPE[dataset_type]

        url = f"{self.DOWNLOAD_BASE_URL}/{dataset_filename}"
        destination_filepath = destination_dir / dataset_filename
        print(f"[LOG]destination_filepath: {destination_filepath}")

        if (not is_force_download) and (destination_filepath.exists()):
            print(f"Bypassing download of already-downloaded file: {str(destination_filepath)}")
            return destination_filepath

        print(f"[LOG]Downloading file {os.path.basename(url)} to {str(destination_filepath)}")
        with request.urlopen(url) as response, destination_filepath.open("wb") as out_file:
            data = response.read()
            out_file.write(data)
        assert destination_filepath.is_file()

        nBytes = destination_filepath.stat().st_size
        print(f"...done, {nBytes} bytes.")
        return destination_filepath
