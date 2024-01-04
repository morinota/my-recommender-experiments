from typing import Optional
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


# コードを書く際はタイプヒントを忘れずに
class AdditiveAttention(nn.Module):
    def __init__(self, emb_size: int, hidden_size: int) -> None:
        super(AdditiveAttention, self).__init__()
        self.fc_layer1 = nn.Linear(emb_size, hidden_size, bias=True)  # 論文中のVとv
        self.fc_layer2 = nn.Linear(hidden_size, 1, bias=False)  # 論文中のq

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, emb_size)
            attn_mask: (batch_size, seq_len)
        Returns:
            (shape) batch_size, emb_size
        """
        # (shape): batch_size, seq_len
        a = self.fc_layer2(nn.Tanh()(self.fc_layer1(x)))

        if attn_mask:
            a = a * attn_mask.unsqueeze(2)

        # (shape): batch_size, seq_len
        alpha = torch.exp(a) / (torch.sum(torch.exp(a), dim=1, keepdim=True) + 1e-8)

        # (shape): batch_size, emb_size
        return torch.bmm(x.permute(0, 2, 1), alpha).squeeze(dim=-1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_k: int,
        d_v: int,
    ) -> None:
        super(MultiHeadSelfAttention, self).__init__()
