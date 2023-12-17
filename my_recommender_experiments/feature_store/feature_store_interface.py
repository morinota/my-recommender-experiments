import abc
from typing import Any


class FeatureStoreInterface(abc.ABC):
    @abc.abstractmethod
    def read_features(self, feature_name: str) -> dict[str, Any]:
        pass

    @abc.abstractmethod
    def write_features(
        self,
        feature_name: str,
        features: list[dict[str, Any]],
    ) -> None:
        pass

    @abc.abstractmethod
    def update_features(
        self,
        feature_name: str,
        features: list[dict[str, Any]],
    ) -> None:
        pass

    @abc.abstractmethod
    def delete_features(self, feature_name: str) -> None:
        pass
