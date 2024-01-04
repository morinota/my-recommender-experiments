from pathlib import Path
import tempfile
from feature_pipelines.dataset_preparer import DatasetPreparer


def test_already_existing_mind_dataset_is_just_converted_to_atomic_files():
    # Arrange
    dataset_type = "mind_training_small"
    destination_dir = Path("./feature_store/data/")
    if not destination_dir.joinpath("MINDsmall_train.zip").exists():
        raise Exception("MINDDataset is not existing. please download it before testing.")
    # MINDsmall_train.zip以外のファイルを削除しておく
    for filepath in destination_dir.iterdir():
        if filepath.name != "MINDsmall_train.zip":
            filepath.unlink()
    sut = DatasetPreparer()

    # Act
    atomic_files = sut.run(dataset_type, destination_dir, False)

    # Assert
    atomic_files_expected = [
        destination_dir / "mind_training_small.inter",
        destination_dir / "mind_training_small.item",
    ]
    for filepath in atomic_files:
        assert filepath.exists()
        assert filepath in atomic_files_expected
