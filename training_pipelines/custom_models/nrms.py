"""
NRMS
################################################
Reference:
    Wu et al. "Neural News Recommendation with Multi-Head Self-Attention." in EMNLP 2019.
Reference:
    https://github.com/yflyl613/NewsRecommendation
"""

from turtle import pos
from typing import Optional
from sympy import Abs
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender, AbstractRecommender
from recbole.config import Config
from recbole.data.interaction import Interaction
from recbole.data.dataset import Dataset
from custom_models.layers import AdditiveAttention, MultiHeadSelfAttention


# 参考: https://recbole.io/docs/developer_guide/customize_models.html
class NRMS(AbstractRecommender):
    def __init__(self, config: Config, dataset: Dataset) -> None:
        super(NRMS, self).__init__(config, dataset)
        self.config = config
        pretrained_word_embedding = torch.from_numpy().float()
        word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding, freeze=True)
        self.news_encoder = NewsEncoder(config, word_embedding)
        self.user_encoder = UserEncoder(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def calculate_loss(self, interaction: Interaction) -> Tensor:
        """
        Args:
            interaction: Interaction (batch_size, history_length, num_words_per_news)
            データローダーから渡される
        Returns:
            loss: Tensor (1)
        """
        reading_histories = interaction[self.reading_history]  # (batch_size, history_length, num_words_per_news)
        candidates = interaction[self.news_candidate]  # (batch_size, 1+K, num_words_per_news)
        labels = interaction[self.label]  # (batch_size, 1+K)

        candidate_news_vecs = self.news_encoder(candidates)  # (batch_size, 1+K, news_vec_dim)

        history_news_vecs = self.news_encoder(reading_histories)  # (batch_size, history_length, news_vec_dim)
        user_vectors = self.user_encoder(history_news_vecs)  # (batch_size, news_vec_dim)

        scores = torch.bmm(candidate_news_vecs, user_vectors)  # (batch_size, 1+K, 1)

        return self.loss_fn(scores, labels)

    def predict(self, interaction: Interaction) -> Tensor:
        reading_histories = interaction[self.reading_history]  # (batch_size, history_length, num_words_per_news)
        candidates = interaction[self.news_candidate]  # (batch_size, num_words_per_news)

        user_vectors = self.user_encoder(self.news_encoder(reading_histories))  # (batch_size, news_vec_dim)
        candidate_news_vectors = self.news_encoder(candidates)  # (batch_size, news_vec_dim)

        return torch.bmm(user_vectors, candidate_news_vectors)  # (batch_size, 1)


class NewsEncoder(nn.Module):
    def __init__(self, config: Config, word_embedding: nn.Embedding) -> None:
        super().__init__()
        self.config = config
        self.word_embedding = word_embedding
        self.dropout_prob = config["dropout_prob"]  # 0.2

        self.dim_per_head = config["embedding_size"] // config["num_attention_heads"]
        assert config["embedding_size"] == self.dim_per_head * config["num_attention_heads"]  # ちょうど割り切れてほしいってことか。

        self.multi_head_self_attn = MultiHeadSelfAttention(
            config["embedding_size"],
            config["num_attention_heads"],
            self.dim_per_head,
            self.dim_per_head,
        )
        self.additive_attn = AdditiveAttention(
            config["embedding_size"],
            config["news_query_vector_dim"],
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, word_num_per_news) (tokenizeは完了済み)
        Returns:
            (shape): batch_size, embedding_size
        """
        # word embedding layer
        word_vecs = F.dropout(
            self.word_embedding(x),
            p=self.dropout_prob,
            training=self.training,
        )  # (shape): batch_size, word_num_per_news, word_dim

        # word-level multi-head self-attention layer
        multihead_word_vecs = F.dropout(
            self.multi_head_self_attn(word_vecs, word_vecs, word_vecs),
            p=self.dropout_prob,
            training=self.training,
        )  # (shape): batch_size, word_num_per_news, embedding_size

        # word-level additive attention layer
        return self.additive_attn(multihead_word_vecs)  # (shape): batch_size, embedding_size


class UserEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.dim_per_head = config["embedding_size"] // config["num_attention_heads"]
        assert config["embedding_size"] == self.dim_per_head * config["num_attention_heads"]

        self.multi_head_self_attn = MultiHeadSelfAttention(
            config["embedding_size"],
            config["num_attention_heads"],
            self.dim_per_head,
            self.dim_per_head,
        )
        self.additive_attn = AdditiveAttention(
            config["embedding_size"],
            config["news_query_vector_dim"],
        )
        # reading histroyの欠損をpaddingするためのベクトルを初期化
        self.padding_vector: Tensor = nn.Parameter(torch.empty(1, config["embedding_size"]).uniform_(-1, 1)).type(
            torch.FloatTensor
        )  # (shape): 1, embedding_size

    def forward(self, news_vectors: Tensor) -> Tensor:
        """
        Args:
            news_vectors: (batch_size, history_length, news_dim)
        Returns:
            (shape): batch_size, news_dim
        """
        batch_size = news_vectors.shape[0]

        padding_doc = self.padding_vector.unsqueeze(0).expand(
            batch_size, self.config["history_length"], -1
        )  # (shape): batch_size, history_length, news_dim
        padded_news_vecs = news_vectors + padding_doc  # (shape): batch_size, history_length, news_dim

        multi_head_news_vecs = self.multi_head_self_attn(padded_news_vecs, padded_news_vecs, padded_news_vecs)

        return self.additive_attn(multi_head_news_vecs)
