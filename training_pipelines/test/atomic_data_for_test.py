# テスト用のatomicデータをtempディレクトリに作成し、そのパスを返すfixture
from pathlib import Path
import tempfile

import pandas as pd


def get_test_data_path(dataset_name: str) -> Path:
    """
    テスト用のatomic fileを作成する。

    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        tmp_dir_path.joinpath(f"{dataset_name}.inter").touch()
        tmp_dir_path.joinpath(f"{dataset_name}.item").touch()
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
        item_df.to_csv(tmp_dir_path.joinpath(f"{dataset_name}.item"), sep="\t")
        inter_df.to_csv(tmp_dir_path.joinpath(f"{dataset_name}.inter"), sep="\t")

    return tmp_dir_path
