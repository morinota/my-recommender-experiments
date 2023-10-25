from pathlib import Path
import unittest


from recommender_experiments.model.data_flow.convert_raw_input_to_atomic import ConvertRawInputToAtomicTask
import shutil


class TestConvertRawInputToAtomicTask(unittest.TestCase):
    def test_run(self):
        # Arrange
        zip_path = Path(r"C:\Users\Masat\src\my-recommender-experiments\my_recommender_experiments\MINDsmall_train.zip")
        output_path = zip_path.parent / "converted"
        dataset_kind = "validation_small"
        sut = ConvertRawInputToAtomicTask()

        # Act
        atomic_files = sut._run(zip_path, output_path, dataset_kind)

        # Assert
        assert atomic_files[0] == output_path / f"MIND_{dataset_kind}_behaviors.inter"
        assert atomic_files[1] == output_path / f"MIND_{dataset_kind}_news.item"
        assert atomic_files[2] == output_path / f"MIND_{dataset_kind}_entity_embedding.item"
        assert atomic_files[3] == output_path / f"MIND_{dataset_kind}_relation_embedding.item"
        shutil.rmtree(output_path)
