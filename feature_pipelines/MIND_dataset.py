from dataclasses import dataclass
from pathlib import Path
import tempfile
import zipfile

import pandas as pd

# from feature_pipelines.base_dataset import RawDatasetInterface
from impression_separator import ImpresionsSeparator


@dataclass
class MINDDataset:
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

    behaviors_feature_type_by_name = {
        "impression_id": "token",
        "user_id": "token",
        "time": "float",
        "history": "token_seq",
        "news_id": "token",
        "label": "float",
    }
    news_feature_type_by_name = {
        "news_id": "token",
        "category": "token",
        "subcategory": "token",
        "title": "token_seq",
        "abstract": "token_seq",
        "url": "token",
        "title_entities": "token_seq",
        "abstract_entities": "token_seq",
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
