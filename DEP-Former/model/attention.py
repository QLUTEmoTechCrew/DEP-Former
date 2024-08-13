import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrossAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout, mix=False):
        super(CrossAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.mix = mix

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)

        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_Face, input_Voice, input_Adapter, attn_mask):
        residual, batch_size = input_Face.clone(), input_Face.size(0)
        Q = (
            self.W_Q(input_Face)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.W_K(input_Voice)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.W_V(input_Adapter)
            .view(batch_size, -1, self.n_heads, self.d_v)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k
        )

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            scores.masked_fill_(
                attn_mask.type(torch.bool),
                torch.from_numpy(np.array(-np.inf)).to(device),
            )

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2)
        if self.mix:
            context = context.transpose(1, 2)
        context = context.reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.dropout(self.fc(context))
        return self.norm(output + residual)


class ProbAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, c, dropout, index, mix=False):
        super(ProbAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.c = c
        self.mix = mix

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)

        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.ind = index

        self.alpha = nn.Sequential(nn.Linear(3072, 1), nn.Sigmoid())

    def _prob_QK(self, Q, K, sample_k, n_top, ind):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E).clone()
        index_sample = torch.randint(
            0, L_K, (L_Q, sample_k)
        )
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
        M = torch.max(Q_K_sample, dim=-1).values - torch.mean(Q_K_sample, dim=-1)
        M_top_index = torch.topk(M, n_top, dim=-1).indices

        M_top_index_out = 0

        if self.ind == 1 and ind is not None:

            Q_a = Q.reshape(Q.size(0), -1)
            attention_weights = self.alpha(Q_a)
            _, top_idx = torch.topk(attention_weights.squeeze(), int(0.5 * B))
            _, low_idx = torch.topk(attention_weights.squeeze(), int(0.5 * B), largest=False)
            index = torch.cat([top_idx, low_idx], dim=0)
            M_top_index_top = M_top_index[top_idx, :, :]
            ind_low = ind[low_idx, :, :]

            Q_sample_low = Q[
                           torch.arange(B)[top_idx, None, None],
                           torch.arange(H)[None, :, None],
                           ind_low,
                           :,
                           ]
            Q_sample_top = Q[
                           torch.arange(B)[low_idx, None, None],
                           torch.arange(H)[None, :, None],
                           M_top_index_top,
                           :,
                           ]
            Q_sample = torch.cat([Q_sample_top, Q_sample_low], dim=0)
            Q_sample = Q_sample[index]

        else:
            Q_sample = Q[
                       torch.arange(B)[:, None, None],
                       torch.arange(H)[None, :, None],
                       M_top_index,
                       :,
                       ]

        return Q_sample, M_top_index, M_top_index_out

    def _get_initial_context(self, V, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if attn_mask is None:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, D).clone()
        else:
            contex = V.cumsum(dim=-2)  # 累积和
        return contex

    def forward(self, input_Q, input_K, input_V, attn_mask, ind=None):
        residual, batch_size = input_Q.clone(), input_Q.size(0)
        L_K, L_Q = input_K.size(1), input_Q.size(1)

        u_k = min(int(self.c * np.log(L_K)), L_Q)
        u_q = min(int(self.c * np.log(L_Q)), L_Q)

        Q = (
            self.W_Q(input_Q)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.W_K(input_K)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.W_V(input_V)
            .view(batch_size, -1, self.n_heads, self.d_v)
            .transpose(1, 2)
        )

        Q_sample, index, M_top_index_out = self._prob_QK(Q, K, sample_k=u_k, n_top=u_q, ind=ind)
        scores = torch.matmul(Q_sample, K.transpose(-1, -2)) / np.sqrt(self.d_k)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

            attn_mask = attn_mask[
                        torch.arange(batch_size)[:, None, None],
                        torch.arange(self.n_heads)[None, :, None],
                        index,
                        :,
                        ]
            scores.masked_fill_(
                attn_mask.type(torch.bool),
                torch.from_numpy(np.array(-np.inf)).to(device),
            )

        attn = nn.Softmax(dim=-1)(scores)
        values = torch.matmul(attn, V)

        context = self._get_initial_context(V, L_Q, attn_mask)
        context[
        torch.arange(batch_size)[:, None, None],
        torch.arange(self.n_heads)[None, :, None],
        index,
        :,
        ] = values

        context = context.transpose(1, 2)
        if self.mix:
            context = context.transpose(1, 2)
        context = context.reshape(batch_size, -1, self.n_heads * self.d_v)

        output = self.dropout(self.fc(context))
        return self.norm(output + residual), index, M_top_index_out


