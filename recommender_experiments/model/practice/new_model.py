from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType
from recbole.config import Config
from recbole.data.dataset import Dataset
import torch
import torch.nn as nn
from recbole.model.loss import BPRLoss  # pytorchで実装されてる。なのでカスタムする時はpytorchと同じやり方でOKそう。
from recbole.model.init import xavier_normal_initialization
from recbole.data.interaction import Interaction

class NewModel(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config: dict, dataset: Dataset) -> None:
        super().__init__(config, dataset)

        self.n_users = dataset.user_num
        self.n_items = dataset.item_num

        self.embedding_size: int = config["embedding_size"]

        self.user_emedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_emedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss_function = BPRLoss()

        # parameter initialization
        self.apply(xavier_normal_initialization)

    def calculate_loss(self, interaction:Interaction):
        user = interaction[]
