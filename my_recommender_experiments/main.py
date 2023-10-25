import argparse
import logging
from pathlib import Path
import sys
import luigi
import numpy as np
import gokart


from recommender_experiments import PipelineTask

CURRENT_DIR = Path("/kaggle/working")


def main() -> int:
    parser = argparse.ArgumentParser()  # コマンドライン引数を定義し、解析する為のparser
    parser.add_argument(
        "--submit",  # --submitオプションを定義
        action="store_true",  # オプションが指定された場合にargs.submitにTrueが入る
    )
    args = parser.parse_args()  # コマンドライン引数をparse(解析)する

    # gokart.add_config(str(CURRENT_DIR.joinpath("my_recommender_experiments/conf/param.ini")))
    gokart.add_config("conf/param.ini")
    # gokart.build(
    #     task=recommender_experiments.Main(),
    #     # log_level=logging.DEBUG,
    # )
    destination_dir = Path("./")
    task = PipelineTask(destination_dir=destination_dir)
    gokart.build(task, log_level=logging.DEBUG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
