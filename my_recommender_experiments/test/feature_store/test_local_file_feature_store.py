from pathlib import Path
import pytest
import tempfile
from feature_store.local_file_feature_store import LocalFileFeatureStore


def test_write_features():
    feature_store_dir = Path(tempfile.TemporaryDirectory().name)
    feature_name = "test_feature"
    features = [
        {"id": "0", "feature": "value0"},
        {"id": "1", "feature": "value1"},
    ]
    sut = LocalFileFeatureStore(feature_store_dir)

    sut.write_features(feature_name, features)

    feature_file = feature_store_dir.joinpath(f"{feature_name}.json")
    assert feature_file.exists()


def test_read_features():
    feature_store_dir = Path(tempfile.TemporaryDirectory().name)
    sut = LocalFileFeatureStore(feature_store_dir)

    feature_name = "test_feature"
    features = [
        {"id": "0", "feature": "value0"},
        {"id": "1", "feature": "value1"},
    ]
    sut.write_features(feature_name, features)

    read_data_actual = sut.read_features(feature_name)
    read_data_expected = [
        {"id": "0", "feature": "value0"},
        {"id": "1", "feature": "value1"},
    ]
    assert read_data_expected == read_data_actual


def test_update_features():
    # Arrange
    feature_store_dir = Path(tempfile.TemporaryDirectory().name)
    sut = LocalFileFeatureStore(feature_store_dir)
    feature_name = "test_feature"
    initial_features = [
        {"id": "0", "feature": "value0"},
        {"id": "1", "feature": "value1"},
    ]
    sut.write_features(feature_name, initial_features)
    incremented_features = [
        {"id": "2", "feature": "value2"},
        {"id": "3", "feature": "value3"},
    ]

    sut.update_features(feature_name, incremented_features)

    updated_features_actual = sut.read_features(feature_name)
    updated_features_expected = [
        {"id": "0", "feature": "value0"},
        {"id": "1", "feature": "value1"},
        {"id": "2", "feature": "value2"},
        {"id": "3", "feature": "value3"},
    ]
    assert updated_features_actual == updated_features_expected


def test_delete_features():
    feature_store_dir = Path(tempfile.TemporaryDirectory().name)
    feature_name = "test_feature"
    features = [
        {"id": "0", "feature": "value0"},
        {"id": "1", "feature": "value1"},
    ]
    sut = LocalFileFeatureStore(feature_store_dir)
    sut.write_features(feature_name, features)

    sut.delete_features(feature_name)

    feature_file = feature_store_dir.joinpath(f"{feature_name}.json")
    assert not feature_file.exists()
