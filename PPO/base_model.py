import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import math

"""
Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.permute(pe, (1,0,2))
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class BaseTokens(nn.Module):
    def __init__(self, device,
                    max_len,
                    num_tokens,
                    dim=64,
                    nhead=8,
                    num_encoders=2,
                    num_decoders=2,
                    d_feedforward=1024,
                    dropout=0.1,
                    batch_first=True):
        super(BaseTokens, self).__init__()
        self.max_len = max_len
        self.device = device
        self.tokens = num_tokens
        self.dim = dim
        ##Create encoder layers##
        self.src_emb = nn.Embedding(num_tokens, dim) 
        self.tgt_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = PositionalEncoding(dim, max_len=max_len)
        ##Create Transformer##
        self.transformer = nn.Transformer(d_model=dim, nhead=nhead,num_encoder_layers=num_encoders,     \
                                         num_decoder_layers=num_decoders, dim_feedforward=d_feedforward, \
                                         dropout=dropout, batch_first=batch_first)
        ##Create Final Linear Layer##
        self.policy_head = nn.Sequential(*[nn.Linear(dim, num_tokens), nn.ReLU(), nn.Linear(num_tokens, num_tokens)])
        self.value_head  = nn.Sequential(*[nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)])

    def get_src_pad_mask(src, pad_idx):
        return src == pad_idx

    def forward(self, src, tgt, pad_idx, value=False):
        src_in = self.pos_emb(self.src_emb(src))
        tgt_in = self.pos_emb(self.tgt_emb(tgt))
        casual_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1])
        pad_mask = Base.get_src_pad_mask(src, pad_idx)
        trans_out = self.transformer(src=src_in, tgt=tgt_in, tgt_mask=casual_mask, \
                    src_key_padding_mask=pad_mask)
        if value:
            return self.policy_head(trans_out), self.value_head(trans_out)

        return self.policy_head(trans_out)

class BaseVectors(nn.Module):
    def __init__(self, device,
                    max_len,
                    num_tokens,
                    src_dim,
                    tgt_dim,
                    dim=64,
                    nhead=8,
                    num_encoders=2,
                    num_decoders=2,
                    d_feedforward=1024,
                    dropout=0.1,
                    src_linear_embedding=False,
                    tgt_linear_embedding=False,
                    batch_first=True):

        super(BaseVectors, self).__init__()
        assert (not src_linear_embedding or dim==src_dim), "Invalid Configuration src emb"

        """Store Relevant Variables"""
        self.max_len = max_len
        self.device = device
        self.dim = dim
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.num_tokens = num_tokens #num_states
        self.src_linear_embedding = src_linear_embedding 
        self.tgt_linear_embedding = tgt_linear_embedding 
        
        

        """Initialize Linear Embeddings"""
        if src_linear_embedding:
            self.src_emb = nn.Linear(src_dim, dim)
        if tgt_linear_embedding:
            self.tgt_emb = nn.Linear(tgt_dim, dim)


        """Initialize Model Components"""
        self.context = torch.nn.Parameter(torch.randn(1,dim))
        self.pos_emb = PositionalEncoding(dim, max_len=max_len)
        self.transformer = nn.Transformer(d_model=dim, nhead=nhead,num_encoder_layers=num_encoders,     \
                                         num_decoder_layers=num_decoders, dim_feedforward=d_feedforward, \
                                         dropout=dropout, batch_first=batch_first)
        ##Create Final Linear Layer##
        self.policy_head = nn.Sequential(*[nn.Linear(dim, num_tokens), nn.ReLU(), nn.Linear(num_tokens, num_tokens)])
        self.value_head  = nn.Sequential(*[nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)])

    def forward(self, src, tgt, value=False):
        N = src.shape[0]

        src = src.to(torch.float32)

        if self.src_linear_embedding:
            src_in = self.pos_emb(self.src_emb(src))
        else: 
            src_in = src

        context = self.context.expand(N, 1, self.dim)

        if tgt is None:
            tgt_in = context
        elif self.tgt_linear_embedding:
            tgt = tgt.to(torch.float32)
            tgt_in = self.pos_emb(self.tgt_emb(tgt))
            tgt_in  = torch.cat((context, tgt_in), dim=1)
        else:
            tgt = tgt.to(torch.float32)
            tgt_in  = torch.cat((context, tgt), dim=1)


        casual_mask = self.transformer.generate_square_subsequent_mask(tgt_in.shape[1])
        trans_out = self.transformer(src=src_in, tgt=tgt_in, tgt_mask=casual_mask)
        if value:
            return self.policy_head(trans_out), self.value_head(trans_out)

        return self.policy_head(trans_out)
