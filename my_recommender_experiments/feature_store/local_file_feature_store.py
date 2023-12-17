import json
from pathlib import Path
from typing import Any
from feature_store.feature_store_interface import FeatureStoreInterface


class LocalFileFeatureStore(FeatureStoreInterface):
    """データの保存形式はjson lineを想定"""

    def __init__(self, directory_path: Path) -> None:
        if not directory_path.exists():
            directory_path.mkdir(parents=True)
        self.directory_path = directory_path

    def read_features(self, feature_name: str) -> list[dict[str, Any]]:
        file_path = self.directory_path / f"{feature_name}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Feature {feature_name} not found")
        with open(file_path) as f:
            return json.load(f)

    def write_features(
        self,
        feature_name: str,
        features: list[dict[str, Any]],
        is_force: bool = False,
    ) -> None:
        file_path = self.directory_path / f"{feature_name}.json"
        if file_path.exists() and not is_force:
            raise FileExistsError(f"Feature {feature_name} already exists")
        with open(file_path, "w") as f:
            json.dump(features, f)

    def update_features(
        self,
        feature_name: str,
        features: list[dict[str, Any]],
    ) -> None:
        file_path = self.directory_path / f"{feature_name}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Feature {feature_name} not found")

        existing_features = self.read_features(feature_name)
        existing_features.extend(features)
        self.write_features(feature_name, existing_features, is_force=True)

    def delete_features(self, feature_name: str) -> None:
        file_path = self.directory_path / f"{feature_name}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Feature {feature_name} not found")
        file_path.unlink()
