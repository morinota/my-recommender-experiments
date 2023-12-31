import abc
import dataclasses
from os import wait
from random import shuffle
from tkinter import wantobjects
from typing import Any


@dataclasses.dataclass(frozen=True)
class AbstractConfig(abc.ABC):
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class EnvironmentSettings(AbstractConfig):
    gpu_id: str = "0"
    worker: int = 0
    seed: int = 2020
    state: str = "INFO"  # ログレベル
    encoding: str = "utf-8"
    repproducibility: bool = True  # 実験結果を再現可能にするか否か
    data_path: str = "atomic_dataset/"
    checkpoint_dir: str = "save/checkpoints/"  # データセットや学習するモデルパラメータの保存先
    show_progress: bool = True
    save_dataset: bool = False
    dataset_save_path: str = "save/dataset/"
    save_dataloader: bool = False
    dataloaders_save_path: str = "save/dataloader/"
    log_wandb: bool = False
    wandb_project: str = "recbole"
    shuffle: bool = True


@dataclasses.dataclass(frozen=True)
class DataSettings(AbstractConfig):
    # Atomic File Format(ここで指定さえすればcsvでもOK)
    field_separator: str = "\t"
    seq_separator: str = " "
    # Basic Information
    ## Common Features
    USER_ID_FIELD: str = "user_id"
    ITEM_ID_FIELD: str = "item_id"
    RATING_FIELD: str = None
    seq_len: dict[str, int] = dataclasses.field(default_factory=lambda: {"history": 20})
    # Selecvitevly Loading
    load_col: dict[str, list[str]] = dataclasses.field(
        default_factory=lambda: {
            "inter": ["user_id", "item_id", "history", "timestamp"],
            "item": ["item_id", "title", "category", "subcategory"],
        }
    )
    unuserd_col: dict[str, list[str]] = dataclasses.field(
        default_factory=lambda: {
            "inter": ["timestamp"],
        }
    )


@dataclasses.dataclass(frozen=True)
class TrainingSettings(AbstractConfig):
    epochs: int = 10
    train_batch_size: int = 512
    learner: str = "adam"
    learning_rate: float = 0.001
    training_neg_sample_args: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "distribution": "uniform",
            "sample_num": 1,
            "dynamic": False,
            "candidate_num": 0,
        }
    )
    eval_step: int = 1
    stopping_step: int = 10
    clip_grad_norm: Any = None
    loss_decimal_place: int = 4
    weight_decay: float = 0.0
    require_power: bool = False
    enable_amp: bool = False
    enable_scaler: bool = False


@dataclasses.dataclass(frozen=True)
class EvaluationSettings(AbstractConfig):
    # 学習・検証・テストデータのグルーピング戦略の設定
    eval_args: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "group_by": "user",  # interのデータをどのようにグループ化するか。user or None. userを指定した場合、各userのinteraction内でデータを分割する。
            "order": "RO",  # RO: random order, TO: time order
            "split": {
                "RS": [0.8, 0.1, 0.1]
            },  # interのsplit戦略。RS: ratio-based splitting, LS: leave-one-out splitting。RSの場合、[train_ratio, valid_ratio, test_ratio]をlist[比率]で指定する。LSの場合、valueを['valid_and_test', 'valid_only', 'test_only']の中から一つを選ぶ
            "mode": {"valid": "full", "test": "full"},  # validとtestでモデルを評価する際のdata rangeを指定する。
            # サポートしてるmodeは4種:[full'、'unixxx'、'popxxx'、'labeled']。前3種は、implicit feedback用。'labeled'はexplicit feedback用。
            # 'full'は全てのitem集合を評価に使う。
            # 'uninxx'は、テスト集合のpositive sampleに対して、xxx個のnegative sampleを一様ランダムにサンプリングし、これらのデータを評価に使う。
            # 'popxxx'は、テスト集合のpositive sampleに対して、xxx個のnegative sampleを人気順にサンプリングし、これらのデータを評価に使う。
            # 異なるフェーズで特定のmodeを指定できる。
        }
    )
    repeatable: bool = False  # 繰り返し推薦が可能なケースか否か。
    metrics: list[str] = dataclasses.field(
        default_factory=lambda: ["Recall", "MRR", "NDCG", "Hit", "Precision"]
    )  # 評価指標の設定
    topk: int = 10  # 評価指標のtopkの設定
    valid_metric: str = "MRR@10"  # early stopping用のmetric。1つのみ指定可能。
    eval_batch_size: int = 512  # 評価時のbatch size
    metric_decimal_place: int = 4  # 評価指標の小数点以下の桁数
