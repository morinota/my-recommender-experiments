from pathlib import Path
import shutil
import gokart
import requests
import pandas as pd


class Step50DownloadDatasetTask(gokart.TaskOnKart):
    # 特に依存するタスクはないので、reqiures()はなく、run()のみ実装している。
    def run(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip"
        filename = Path("NewsAggregatorDataset.zip")
        if not filename.exists():
            data = requests.get(url).content
            with open(filename, mode="wb") as f:
                f.write(data)
            outdirname = "data/" + str(filename).replace(
                ".zip",
                "",
            )
            shutil.unpack_archive(filename, outdirname)

        colnames = ["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"]
        df = pd.read_csv(
            "data/NewsAggregatorDataset/newsCorpora.csv", header=None, names=colnames, sep="\t", index_col="ID"
        )
        self.dump(df)  # ダンプする事で次のタスクで使う事ができる。
