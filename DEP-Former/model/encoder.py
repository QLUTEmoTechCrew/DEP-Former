import torch
import torch.nn.functional as F
from torch import nn

from model.attention import ProbAttention

from model.embed import DataEmbedding


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=(3,),
            padding=(1,),
            stride=(1,),
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        else:
            x = x
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, c, dropout, index):
        super(EncoderLayer, self).__init__()
        self.attention = ProbAttention(d_k, d_v, d_model, n_heads, c, dropout, index, mix=False)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x, ind=None, attn_mask=None):
        x, index, M_top_index_out = self.attention(x, x, x, attn_mask=attn_mask, ind=ind)
        residual = x.clone()
        x = self.dropout(self.activation(self.conv1(x.permute(0, 2, 1))))
        y = self.dropout(self.conv2(x).permute(0, 2, 1))
        return self.norm(residual + y), index, M_top_index_out


class EncoderLayer_fv(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, c, dropout, index):
        super(EncoderLayer_fv, self).__init__()
        self.attention = ProbAttention(d_k, d_v, d_model, n_heads, c, dropout, index, mix=False)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x, ind=None, attn_mask=None):
        x, index, M_top_index_out = self.attention(x, x, x, attn_mask=attn_mask, ind=ind)
        residual = x.clone()
        x = self.dropout(self.activation(self.conv1(x.permute(0, 2, 1))))
        y = self.dropout(self.conv2(x).permute(0, 2, 1))
        return self.norm(residual + y)


class Encoder(nn.Module):
    def __init__(
            self,
            d_k,
            d_v,
            d_model,
            d_ff,
            n_heads,
            n_layer,
            n_stack,
            d_feature,
            d_mark,
            dropout,
            c,
            index,
    ):
        super(Encoder, self).__init__()

        self.embedding = DataEmbedding(d_feature, d_mark, d_model, dropout)

        self.stacks = nn.ModuleList()
        for i in range(n_stack):
            stack = nn.Sequential()
            stack.add_module(
                "elayer" + str(i) + "0",
                EncoderLayer(d_k, d_v, d_model, d_ff, n_heads, c, dropout, index),
            )

            for j in range(n_layer - i - 1):
                stack.add_module("clayer" + str(i) + str(j + 1), ConvLayer(d_model))
                stack.add_module(
                    "elayer" + str(i) + str(j + 1),
                    EncoderLayer(d_k, d_v, d_model, d_ff, n_heads, c, dropout, index),
                )

            self.stacks.append(stack)
        self.norm = nn.LayerNorm(d_model)

        self.index = index
        self.en_fv = EncoderLayer_fv(d_k, d_v, d_model, d_ff, n_heads, c, dropout, index)

    def forward(self, enc_x, ind=None):
        x = self.embedding(enc_x)
        out = []
        for i, stack in enumerate(self.stacks):
            inp_len = x.shape[1] // (2 ** i)
            y = x[:, -inp_len:, :]
            y, index, M_top_index_out = stack(y)
            y = self.norm(y)

            if self.index == 1:
                y1 = self.en_fv(y, ind)
                y1 = self.norm(y1)
                y = y + y1
            out.append(y)
        out = torch.cat(out, -2)

        return out, index, M_top_index_out
