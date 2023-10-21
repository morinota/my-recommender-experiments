import logging
from pathlib import Path
import luigi
import numpy as np
import gokart


import recommender_experiments


if __name__ == "__main__":
    CURRENT_DIR = Path("/kaggle/working")
    # gokart.add_config(str(CURRENT_DIR.joinpath("my_recommender_experiments/conf/param.ini")))
    gokart.add_config("conf/param.ini")
    # gokart.build(
    #     task=recommender_experiments.Main(),
    #     # log_level=logging.DEBUG,
    # )
    gokart.run()
