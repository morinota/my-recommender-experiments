from pathlib import Path
from task_interface import TaskInterface


class AtomicFileExporter(TaskInterface):
    def __init__(self) -> None:
        pass

    def run(
        self,
        atomic_data_by_filename: dict[str, str],
        destination_dir: Path,
    ) -> list[Path]:
        exported_paths = []
        # 各ファイル名ごとにデータを書き込み
        for filename, data in atomic_data_by_filename.items():
            file_path = destination_dir / filename

            # ファイルにデータを書き込み
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(data)

            exported_paths.append(file_path)

        return exported_paths
