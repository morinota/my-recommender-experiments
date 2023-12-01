from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Optional
import zipfile
from numpy import int8

import pandas as pd
from recommender_experiments.dataset.base_dataset import RawDatasetInterface


@dataclass
class MINDDataset(RawDatasetInterface):
    # selected feature fields
    # 型について -> https://recbole.io/docs/user_guide/data/atomic_files.html#format
    DATASET_KINDS_CANDIDATES = [
        "training_small",
        "validation_small",
        "training_large",
        "validation_large",
    ]
    RAW_FILES_INFO = {
        "behaviors": {
            "filename": "behaviors.tsv",
            "sep": "\t",
        },
        "news": {
            "filename": "news.tsv",
            "sep": "\t",
        },
        "entity_embeddings": {
            "filename": "entity_embedding.vec",
            "sep": "\t",
        },
        "relation_embeddings": {
            "filename": "relation_embedding.vec",
            "sep": "\t",
        },
    }

    behaviors_fields = {
        0: "impression_id:token",
        1: "user_id:token",
        2: "time:float",
        3: "history:token_seq",
        4: "item_id:token",
        5: "is_tap:float",
    }
    news_fields = {
        0: "id:token",
        1: "category:token",
        2: "subcategory:token",
        3: "title:token_seq",
        4: "abstract:token_seq",
        5: "url:token",
        6: "title_entities:token_seq",
        7: "abstract_entities:token_seq",
    }
    entity_embedding_fields = {
        0: "entity_id:token",
        1: "vector:float_seq",
    }
    relation_embedding_fields = {
        0: "relation_id:token",
        1: "vector:float_seq",
    }

    behaviors: pd.DataFrame
    news: pd.DataFrame
    entity_embeddings: pd.DataFrame
    relation_embeddings: pd.DataFrame

    @classmethod
    def load_from_zip(cls, zip_path: Path) -> "MINDDataset":
        """
        - zip_pathのzipファイル内に、4つのファイルが圧縮されている。
        - zipファイルをtemp directoryにunzipし、4つのファイルをpd.DataFrameとしてメモリに載せ、dataclassの各fieldに載せてdataclassとして初期化する。
        """
        unziped_dir = cls._unzip_to_temp_dir(zip_path)

        behaviors = cls._load_behaviors_data(unziped_dir)
        news = cls._load_news_data(unziped_dir)
        entity_embeddings = cls._load_entity_embedding_data(unziped_dir)
        relation_embeddings = cls._load_relation_embedding_data(unziped_dir)

        return MINDDataset(
            behaviors,
            news,
            entity_embeddings,
            relation_embeddings,
        )

    @classmethod
    def _unzip_to_temp_dir(cls, zip_path: Path) -> Path:
        temp_dir = Path(tempfile.gettempdir()) / "mind"
        temp_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        return temp_dir

    @classmethod
    def _load_behaviors_data(cls, unziped_dir: Path) -> pd.DataFrame:
        behavior_df = pd.read_table(
            unziped_dir / cls.RAW_FILES_INFO["behaviors"]["filename"],
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )
        separator = ImpresionsSeparator()
        return separator.separate(behavior_df, "impressions")

    @classmethod
    def _load_news_data(cls, unziped_dir: Path) -> pd.DataFrame:
        return pd.read_table(
            unziped_dir / cls.RAW_FILES_INFO["news"]["filename"],
            header=None,
            names=["id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"],
        )

    @classmethod
    def _load_entity_embedding_data(cls, unziped_dir: Path) -> pd.DataFrame:
        entity_embedding = pd.read_table(
            unziped_dir / cls.RAW_FILES_INFO["entity_embeddings"]["filename"],
            header=None,
        )
        entity_embedding["vector"] = entity_embedding.iloc[:, 1:101].values.tolist()
        entity_embedding = entity_embedding[[0, "vector"]].rename(columns={0: "entity_id"})
        return entity_embedding

    @classmethod
    def _load_relation_embedding_data(cls, unziped_dir: Path) -> pd.DataFrame:
        relation_embedding = pd.read_table(
            unziped_dir / cls.RAW_FILES_INFO["relation_embeddings"]["filename"],
            header=None,
        )
        relation_embedding["vector"] = relation_embedding.iloc[:, 1:101].values.tolist()
        relation_embedding = relation_embedding[[0, "vector"]].rename(columns={0: "entity_id"})
        return relation_embedding


class ImpresionsSeparator:
    def __init__(self) -> None:
        pass

    def separate(
        self,
        behavior_df: pd.DataFrame,
        impressions_col: str = "impressions",
        separated_col: str = "news_id",
    ) -> pd.DataFrame:
        implicit_feedbacks = []
        for _, row in behavior_df.iterrows():
            user_id = row["user_id"]
            impressions_str = row[impressions_col]
            impressions_list = impressions_str.split()

            for impression in impressions_list:
                news_id, is_interact = impression.split("-")
                if is_interact == "0":
                    continue
                implicit_feedbacks.append({"user_id": user_id, separated_col: news_id})

        return pd.DataFrame(implicit_feedbacks)
