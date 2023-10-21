#!/usr/bin/env python3
import base64
import gzip
from pathlib import Path


def _prepare_to_encode(target_dirs: list[str]) -> list[Path]:
    to_encode: list[Path] = []
    for target_dir in target_dirs:
        to_encode += list(Path(target_dir).glob("**/*.*"))
    to_encode += [Path("setup.py")] + [Path("requirements.txt")]
    to_encode_filtered = [path for path in to_encode if _should_encode(path)]
    print(to_encode_filtered)
    return to_encode_filtered


def _should_encode(path: Path) -> bool:
    if path.is_dir():
        return False
    if path.name.startswith("."):
        return False
    return True


def _encode_file(path: Path) -> str:
    compressed = gzip.compress(path.read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode("utf-8")


def build_script():
    to_encode = _prepare_to_encode(["easy_gold", "my_recommender_experiments"])
    file_data = {str(path): _encode_file(path) for path in to_encode}  # ここでなぜか / が \\になっているので置き換えたい。
    valid_file_data = {key.replace("\\", "/"): value for key, value in file_data.items()}
    template = Path("script_template.py").read_text("utf8")
    Path("build/script.py").write_text(template.replace("{file_data}", str(valid_file_data)), encoding="utf8")


if __name__ == "__main__":
    build_script()
