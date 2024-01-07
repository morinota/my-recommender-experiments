from pathlib import Path
import time
import pandas as pd
from tqdm import tqdm


class MINDTrainConverter:
    SEP = "\t"
    LINEBREAK = "\n"

    def __init__(self, raw_data_dir: Path, atomic_data_dir: Path, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.sep = "\t"

        # input file
        self.input_dir = raw_data_dir / self.dataset_name
        self.item_file = self.input_dir / "news.tsv"
        self.inter_file = self.input_dir / "behaviors.tsv"

        # output file
        atomic_data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = atomic_data_dir / self.dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_item_file = self.output_dir / f"{self.dataset_name}.item"
        self.output_inter_file = self.output_dir/ f"{self.dataset_name}.inter"

        # fields
        self.item_fields = {
            0: "news_id:token",
            1: "category:token",
            2: "sub_category:token",
            3: "title:token_seq",
            4: "summary:token_seq",
            5: "url:token_seq",
            6: "title_entities:token_seq",
            7: "summary_entities:token_seq",
        }
        self.inter_fields = {
            0: "impression_id:token",
            1: "user_id:token",
            2: "timestamp:float",
            3: "history:token_seq",
            4: "news_id:token",
        }

    def convert(self) -> None:
        self.
        self._convert_item()
        self._convert_inter_with_only_positive_record()

        # fileのサイズを出力
        print(f"item file size: {self.output_item_file.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"inter file size: {self.output_inter_file.stat().st_size / 1024 / 1024:.2f} MB")

    def _convert_item(self) -> None:
        """
        Convert item.tsv to item.inter
        """
        # headerカラムを書き込む
        atomic_records = [self.SEP.join(self.item_fields[column] for column in self.item_fields.keys())]

        with self.item_file.open("r", encoding="utf-8") as input_file, self.output_item_file.open(
            "w", encoding="utf-8"
        ) as output_file:
            df = pd.read_table(
                input_file,
                header=None,
                dtype=str,
                names=[
                    "news_id",
                    "category",
                    "subcategory",
                    "title",
                    "abstract",
                    "url",
                    "title_entities",
                    "abstract_entities",
                ],
            )
            # 欠損値を空文字列で埋める
            df = df.fillna("")
            print(df.head())

            for record in df.itertuples():
                features = list(record)[1:]  # tupleからdfのindexを除く

                news_id = features[0].strip("N")  # news_idのprefixを削除
                category = features[1]
                sub_category = features[2]
                title = features[3]
                summary = features[4]
                url = features[5]
                title_entities = features[6]
                summary_entities = features[7]
                atomic_record = self.SEP.join(
                    [news_id, category, sub_category, title, summary, url, title_entities, summary_entities]
                )

                atomic_records.append(atomic_record)

            atomic_data_str = self.LINEBREAK.join(atomic_records)
            output_file.write(atomic_data_str)

    def _convert_inter_with_only_positive_record(self) -> None:
        """
        impression内のpositive recordのみを残すver
        """
        with self.inter_file.open("r") as input_file, self.output_inter_file.open("w") as output_file:
            lines = input_file.readlines()
            # headerカラムを書き込む
            output_file.write("\t".join([self.inter_fields[column] for column in self.inter_fields.keys()]) + "\n")

            # 各行を書き込む
            for line in tqdm(lines):
                # user_id, item_id, rating, timestampに分割
                features = line.split("\t")

                impression_id = features[0]
                user_id = features[1].strip("U")  # user_idのprefixを削除
                timestamp = self._convert_unix_timestamp(features[2])  # timeをunix timestampに変換
                rating_list = features[4].split()  # impressionsカラムの中身を取り出す
                history = features[3]

                for rate in rating_list:
                    item, rating = rate.split("-")  # item_idとratingに分割
                    item = item.strip("N")  # item_idのprefixを削除
                    if rating != "1":
                        continue
                    output_file.write("\t".join([impression_id, user_id, timestamp, history, item]) + "\n")

    def _convert_inter_with_impressions(self) -> None:
        """
        impressionsカラムを残しlabelカラムやnews_idカラムを作らないver
        """
        with self.inter_file.open("r") as input_file, self.output_inter_file.open("w") as output_file:
            lines = input_file.readlines()
            # headerカラムを書き込む
            output_file.write("\t".join([self.inter_fields[column] for column in self.inter_fields.keys()]) + "\n")

            # 各行を書き込む
            for line in tqdm(lines):
                # user_id, item_id, rating, timestampに分割
                line_list = line.split("\t")

                impression_id = line_list[0]
                user_id = line_list[1].strip("U")  # user_idのprefixを削除
                timestamp = self._convert_unix_timestamp(line_list[2])  # timeをunix timestampに変換
                impressions = line_list[4]
                history = line_list[3]

                output_file.write("\t".join([impression_id, user_id, timestamp, impressions, history]) + "\n")

    def _convert_inter_with_positive_and_negative_record(self) -> None:
        """
        Convert behaviors.tsv to inter.inter
        """
        with self.inter_file.open("r") as input_file, self.output_inter_file.open("w") as output_file:
            lines = input_file.readlines()
            # headerカラムを書き込む
            output_file.write("\t".join([self.inter_fields[column] for column in self.inter_fields.keys()]) + "\n")

            # 各行を書き込む
            for line in tqdm(lines):
                # user_id, item_id, rating, timestampに分割
                line_list = line.split("\t")

                user_id = line_list[1].strip("U")  # user_idのprefixを削除
                timestamp = self._convert_unix_timestamp(line_list[2])  # timeをunix timestampに変換
                rating_lst = line_list[4].split()  # impressionsカラムの中身を取り出す
                history = line_list[3]

                for rate in rating_lst:
                    item, rating = rate.split("-")  # item_idとratingに分割
                    item = item.strip("N")  # item_idのprefixを削除
                    output_file.write("\t".join([user_id, item, rating, timestamp, history]) + "\n")

    def _convert_unix_timestamp(self, time_str: str) -> str:
        """
        Convert timestamp from "mm/dd/yyyy hh:mm:ss AM/PM" to unix timestamp
        """
        if "AM" in time_str:
            return str(int(time.mktime(time.strptime(time_str, "%m/%d/%Y %H:%M:%S AM"))))
        else:
            return str(int(time.mktime(time.strptime(time_str, "%m/%d/%Y %H:%M:%S PM"))) + 43200)


if __name__ == "__main__":
    feature_store_dir = Path("./feature_store")
    dataset_name = "mind_training_small"
    input_path = feature_store_dir / "raw_data"
    output_path = feature_store_dir / "atomic_data"

    item_atomic_path = output_path / dataset_name / f"{dataset_name}.item"
    df = pd.read_csv(item_atomic_path, sep="\t", engine="python")
    print(df.head())
