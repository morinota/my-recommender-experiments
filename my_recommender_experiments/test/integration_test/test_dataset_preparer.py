from pathlib import Path
import tempfile
from recommender_experiments.model.data_flow.dataset_preparer import DatasetPreparer


def test_mind_dataset_is_prepared_as_atomic_files_for_recbole() -> None:
    # Arrange
    dataset_type = "validation_small"
    temp_dir = tempfile.TemporaryDirectory()
    destination_dir = Path(temp_dir.name)
    sut = DatasetPreparer()

    # Act
    atomic_files = sut.run(dataset_type, destination_dir)

    # Assert
    # atomic_files_expected = [destination_dir / ""]
    for filepath in atomic_files:
        assert filepath.exists()
