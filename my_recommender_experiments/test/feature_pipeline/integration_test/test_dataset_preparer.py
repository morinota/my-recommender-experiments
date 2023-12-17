from pathlib import Path
import tempfile
from feature_pipelines.dataset_preparer import DatasetPreparer


# def test_mind_dataset_is_downloaded_and_prepared_as_atomic_files() -> None:
#     # Arrange
#     dataset_type = "validation_small"
#     temp_dir = tempfile.TemporaryDirectory()
#     destination_dir = Path(temp_dir.name)
#     sut = DatasetPreparer()

#     # Act
#     atomic_files = sut.run(dataset_type, destination_dir)

#     # Assert
#     atomic_files_expected = [destination_dir / "behavior.inter", destination_dir / "news.item"]
#     for filepath in atomic_files:
#         assert filepath.exists()


def test_already_existing_mind_dataset_is_just_converted_to_atomic_files():
    # Arrange
    dataset_type = "validation_small"
    destination_dir = Path("my_recommender_experiments/")
    if not destination_dir.joinpath("MINDsmall_dev.zip").exists():
        raise Exception("MINDDataset is not existing. please download it before testing.")
    sut = DatasetPreparer()

    # Act
    atomic_files = sut.run(dataset_type, destination_dir, False)

    # Assert
    atomic_files_expected = [
        destination_dir / "behavior.inter",
        destination_dir / "news.item",
        destination_dir / "entity_embeddings.ent",
        destination_dir / "relation_embeddings.rel",
    ]
    for filepath in atomic_files:
        assert filepath.exists()
        assert filepath in atomic_files_expected
