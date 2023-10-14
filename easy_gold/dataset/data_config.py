from pathlib import Path


class MINDPathConfig:
    parent_dir = Path("/kaggle/input/mind-news-dataset")
    news_tsv = parent_dir.joinpath("news.tsv/news.tsv")
    behaviors_tsv = parent_dir.joinpath("MINDsmall_train/behaviors.tsv")
    entity_embedding_vec = parent_dir.joinpath("MINDsmall_train/entity_embedding.vec")
    relation_embedding_vec = parent_dir.joinpath("MINDsmall_train/relation_embedding.vec")
