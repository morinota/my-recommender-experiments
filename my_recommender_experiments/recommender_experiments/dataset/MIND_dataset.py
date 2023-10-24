import os
from pathlib import Path
from typing import Optional
from numpy import int8

import pandas as pd
from recommender_experiments.dataset.base_dataset import BaseDataset


class MINDDataset(BaseDataset):
    BASE_NAME = "MIND"
    DATASET_KINDS_CANDIDATES = [
        "training_small",
        "validation_small",
        "training_large",
        "validation_large",
    ]

    def __init__(self, input_path: Path, output_path: Path, dataset_kind: str) -> None:
        if dataset_kind not in self.DATASET_KINDS_CANDIDATES:
            raise ValueError(f"`dataset_kind` argument need to be in {self.DATASET_KINDS_CANDIDATES}")

        self.input_path = input_path
        self.output_path = output_path
        self.check_output_path()
        self.dataset_name = f"{self.BASE_NAME}_{dataset_kind}"

        # input_path
        self.behavior_file = self.input_path / "behaviors.tsv"
        self.news_file = self.input_path / "news.tsv"
        self.entity_embedding_file = self.input_path / "entity_embedding.vec"
        self.relation_embedding_file = self.input_path / "relation_embedding.vec"
        self.sep = "\t"

        # output_path
        output_files = self._get_output_files()
        self.output_behavior_file = output_files[0]
        self.output_news_file = output_files[1]
        self.output_entity_embedding_file = output_files[2]
        self.output_relation_embedding_file = output_files[3]

        # selected feature fields
        # 型について -> https://recbole.io/docs/user_guide/data/atomic_files.html#format
        self.behaviors_fields = {
            0: "impression_id:token",
            1: "user_id:token",
            2: "time:float",
            3: "history:token_seq",
            4: "item_id:token",
            5: "is_tap:float",
        }
        self.news_fields = {
            0: "id:token",
            1: "category:token",
            2: "subcategory:token",
            3: "title:token_seq",
            4: "abstract:token_seq",
            5: "url:token",
            6: "title_entities:token_seq",
            7: "abstract_entities:token_seq",
        }
        self.entity_embedding_fields = {
            0: "entity_id:token",
            1: "vector:float_seq",
        }
        self.relation_embedding_fields = {
            0: "relation_id:token",
            1: "vector:float_seq",
        }

    def _load_behaviors_data(self) -> pd.DataFrame:
        behavior_df = pd.read_table(
            self.behavior_file,
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )
        return self._expand_each_impression(behavior_df)

    def _load_news_data(self) -> pd.DataFrame:
        return pd.read_table(
            self.news_file,
            header=None,
            names=["id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"],
        )

    def _load_entity_embedding_data(self) -> pd.DataFrame:
        entity_embedding = pd.read_table(self.entity_embedding_file, header=None)
        entity_embedding["vector"] = entity_embedding.iloc[:, 1:101].values.tolist()
        entity_embedding = entity_embedding[[0, "vector"]].rename(columns={0: "entity_id"})
        return entity_embedding

    def _load_relation_embedding_data(self) -> pd.DataFrame:
        relation_embedding = pd.read_table(self.relation_embedding_file, header=None)
        relation_embedding["vector"] = relation_embedding.iloc[:, 1:101].values.tolist()
        relation_embedding = relation_embedding[[0, "vector"]].rename(columns={0: "entity_id"})
        return relation_embedding

    def _get_output_files(self) -> tuple[Path, Path, Path, Path]:
        output_behavior_file = self.output_path / f"{self.dataset_name}_behaviors.inter"
        output_news_file = self.output_path / f"{self.dataset_name}_news.item"
        output_entity_embedding_file = self.output_path / f"{self.dataset_name}_entity_embedding.ent"
        output_relation_embedding_file = self.output_path / f"{self.dataset_name}_relation_embedding.rel"

        return (
            output_behavior_file,
            output_news_file,
            output_entity_embedding_file,
            output_relation_embedding_file,
        )

    def convert_behaviors(self) -> Path:
        try:
            input_behaviors_data = self._load_behaviors_data()
            return self._convert(
                input_behaviors_data,
                self.behaviors_fields,
                self.output_behavior_file,
            )
        except NotImplementedError:
            print("This dataset can't be converted to user file\n")

    def convert_news(self) -> Path:
        input_news_data = self._load_news_data()
        return self._convert(
            input_news_data,
            self.news_fields,
            self.output_news_file,
        )

    def convert_entity_embedding(self) -> Path:
        input_entity_embedding_data = self._load_entity_embedding_data()
        return self._convert(
            input_entity_embedding_data,
            self.entity_embedding_fields,
            self.output_entity_embedding_file,
        )

    def convert_relation_embedding(self) -> Path:
        input_relation_embedding_data = self._load_relation_embedding_data()
        return self._convert(
            input_relation_embedding_data,
            self.relation_embedding_fields,
            self.output_relation_embedding_file,
        )

    def _expand_each_impression(self, behavior_df: pd.DataFrame) -> pd.DataFrame:
        # 各impressionを展開し、新しいDataFrameを作成
        df_except_impressions = behavior_df.drop("impressions", axis=1)
        expanded_df = (
            behavior_df["impressions"]
            .str.split(" ", expand=True)
            .stack()
            .reset_index(level=1, drop=True)
            .to_frame("impression_list")
            .join(df_except_impressions)
        )
        expanded_df[["item_id", "is_tap_str"]] = expanded_df["impression_list"].str.split("-", expand=True)

        expanded_df["is_tap"] = expanded_df["is_tap_str"].astype(int)

        return expanded_df[["impression_id", "user_id", "item_id", "history", "time", "is_tap"]]
