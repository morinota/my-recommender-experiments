"""
NRMS
################################################
Reference:
    Wu et al. "Neural News Recommendation with Multi-Head Self-Attention." in EMNLP 2019.
Reference:
    https://github.com/yflyl613/NewsRecommendation
"""

from typing import Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender, ContextRecommender
from recbole.config import Config
from recbole.data.interaction import Interaction
from recbole.data.dataset import Dataset
from models.layers import AdditiveAttention, MultiHeadSelfAttention


# 参考: https://recbole.io/docs/developer_guide/customize_models.html
class NRMS(ContextRecommender):
    def __init__(self, config: Config, dataset: Dataset) -> None:
        super(NRMS, self).__init__(config, dataset)
        self.config = config
        pretrained_word_embedding = torch.from_numpy().float()
        print("hoge")
        word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding, freeze=True)
        self.news_encoder = NewsEncoder(config, word_embedding)
        self.user_encoder = UserEncoder(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, history: Tensor, history_mask: Tensor, candidate: Tensor, label: Tensor) -> Tensor:
        """
        Args:
            history: (batch_size, history_length, num_words_per_news)
            history_mask: (batch_size, history_length)
            candidate: (batch_size, 1+K, num_words_per_news)
            label: (batch_size, 1+K)
        """
        candidate_news = candidate.reshape(-1, self.config["num_words_per_news"])
        candidate_news_vecs = self.news_encoder(candidate_news).reshape(
            -1, 1 + self.config["npratio"], self.config["news_vec_dim"]
        )

    def calculate_loss(self, interaction: Interaction) -> Tensor:
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]

    def predict(self, interaction: Interaction) -> Tensor:
        pass


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

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: (batch_size, word_num_per_news) (tokenizeは完了済み)
            mask: (batch_size, word_num_per_news)
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
            self.multi_head_self_attn(word_vecs, word_vecs, word_vecs, mask),
            p=self.dropout_prob,
            training=self.training,
        )  # (shape): batch_size, word_num_per_news, embedding_size

        # word-level additive attention layer
        return self.additive_attn(multihead_word_vecs, mask)  # (shape): batch_size, embedding_size


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

    def forward(
        self,
        news_vectors: Tensor,
        log_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            news_vectors: (batch_size, history_length, news_dim)
            log_mask: (batch_size, history_length)
        Returns:
            (shape): batch_size, news_dim
        """
        batch_size = news_vectors.shape[0]
        # log_maskを使って履歴の有効部分のみを使う場合
        if self.config["is_user_log_mask"]:
            # (shape): batch_size, history_length, news_dim
            multi_head_news_vecs = self.multi_head_self_attn(news_vectors, news_vectors, news_vectors, log_mask)
            # (shape): batch_size, news_dim
            user_vec = self.additive_attn(multi_head_news_vecs, log_mask)
        # log_maskを使わずに履歴の全てを使う場合
        else:
            padding_doc = self.padding_vector.unsqueeze(0).expand(batch_size, self.config["history_length"], -1)
            padded_news_vecs = news_vectors * log_mask.unsqueeze(2) + padding_doc * (1 - log_mask).unsqueeze(2)
            multi_head_news_vecs = self.multi_head_self_attn(padded_news_vecs, padded_news_vecs, padded_news_vecs)
            user_vec = self.additive_attn(multi_head_news_vecs)
