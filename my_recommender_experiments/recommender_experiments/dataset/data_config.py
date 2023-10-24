from pathlib import Path


class MINDConfig:
    # refs: https://learn.microsoft.com/ja-jp/azure/open-datasets/dataset-microsoft-news?tabs=azureml-opendatasets
    download_base_url = "https://mind201910small.blob.core.windows.net/release"
    training_small_url = f"{download_base_url}/MINDsmall_train.zip"  # about 50MB
    validation_small_url = f"{download_base_url}/MINDsmall_dev.zip"  # about 30MB
    training_large_url = f"{download_base_url}/MINDlarge_train.zip"
    validation_large_url = f"{download_base_url}/MINDlarge_dev.zip"

    parent_dir = Path("/kaggle/input/mind-news-dataset")
    news_tsv = parent_dir.joinpath("news.tsv/news.tsv")
    behaviors_tsv = parent_dir.joinpath("MINDsmall_train/behaviors.tsv")
    entity_embedding_vec = parent_dir.joinpath("MINDsmall_train/entity_embedding.vec")
    relation_embedding_vec = parent_dir.joinpath("MINDsmall_train/relation_embedding.vec")