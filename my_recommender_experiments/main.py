import argparse
import logging
from pathlib import Path
import sys
import luigi
import numpy as np
import gokart


from recommender_experiments import PipelineTask


def main() -> None:
    parser = argparse.ArgumentParser()  # コマンドライン引数を定義し、解析する為のparser
    parser.add_argument(
        "--submit",  # --submitオプションを定義
        action="store_true",  # オプションが指定された場合にargs.submitにTrueが入る
    )
    args = parser.parse_args()  # コマンドライン引数をparse(解析)する

    all_tasks = PipelineTask()
    all_tasks.run()


if __name__ == "__main__":
    main()
