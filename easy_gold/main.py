from pathlib import Path
from dataset.data_config import MINDPathConfig
import pandas as pd
import gokart
from recbole.quick_start import run_recbole
from recbole.model.sequential_recommender.bert4rec import BERT4Rec
from recbole.config import Config
from recbole import data


def main():
    CURRENT_DIR = Path("/kaggle/working")
    print("Hello world!")
    print(f"{MINDPathConfig.parent_dir}")
    behavior_df = pd.read_csv(MINDPathConfig.behaviors_tsv, sep="\t")
    print(behavior_df.head())

    config = Config(
        model="BPR",
        dataset="ml-1m",
        config_file_list=[CURRENT_DIR.joinpath("easy_gold/config/example.yaml")],
    )
    print(config)
    res = run_recbole(model="BPR", dataset="ml-1m")


if __name__ == "__main__":
    main()
