from dataset.data_config import MINDPathConfig
import pandas as pd
import gokart


def main():
    print("Hello world!")
    print(f"{MINDPathConfig.parent_dir}")
    behavior_df = pd.read_csv(MINDPathConfig.behaviors_tsv, sep="\t")
    print(behavior_df.head())


if __name__ == "__main__":
    main()
