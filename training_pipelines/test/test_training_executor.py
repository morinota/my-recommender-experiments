# サードパーティパッケージ RecBole の複雑性を隠蔽するFacadeクラス的な役割
from pathlib import Path
import tempfile
import pandas as pd


from training_executor import TrainingExecutor


def get_test_data_path() -> Path:
    """
    テスト用のatomic fileを作成する。

    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        tmp_dir_path.joinpath("mind_training_for_test.inter").touch()
        tmp_dir_path.joinpath("mind_training_for_test.item").touch()
    # 各atomic fileに中身を追加する
    item_df = pd.DataFrame(
        {
            "news_id:token": [1, 2],
            "category:token": ["cat1", "cat2"],
            "subcategory:token": ["subcat1", "subcat2"],
        }
    )
    # 5行くらいのinter_dfを作成する
    inter_df = pd.DataFrame(
        {
            "user_id:token": [1, 1, 2, 2, 3],
            "news_id:token": [1, 2, 1, 2, 1],
            "label:float": [0, 1, 1, 0, 0],
        }
    )
    # atomic_fileに書き込む
    item_df.to_csv(tmp_dir_path.joinpath("mind_training_for_test.item"), sep="\t")
    inter_df.to_csv(tmp_dir_path.joinpath("mind_training_for_test.inter"), sep="\t")

    return tmp_dir_path


def test_run_training_recsys_with_recbole_facade() -> None:
    # Arrange
    model = "BPR"
    dataset_name = "mind_training_for_test"
    config_dict = {"data_path": get_test_data_path()}
    sut = TrainingExecutor(model, dataset_name, config_dict)

    # Act
    result = sut.run()

    # Assert
    assert result.is_success == True
    assert result.trained_model_path == Path("./trained_models/NRMS/")
    assert result.log_file_path == Path("./logs/NRMS/")
    assert result.best_valid_result == {"auc": 0.5, "hogehoge": 0.5}
    assert result.test_result == {"auc": 0.5, "hogehoge": 0.5}
