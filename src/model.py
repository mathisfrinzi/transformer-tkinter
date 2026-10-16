# Transformer encoder-decoder fait à la main (seulement nn.Linear et nn.Embedding de PyTorch)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD, SOS, EOS


class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.last = {}  # Q, K, V, scores, poids du dernier passage (pour l'interface)

    def forward(self, q_in, kv_in, mask=None):
        B, Lq, D = q_in.shape
        Lk = kv_in.shape[1]
        Q = self.W_q(q_in).view(B, Lq, self.h, self.d_k).transpose(1, 2)   # (B, h, Lq, dk)
        K = self.W_k(kv_in).view(B, Lk, self.h, self.d_k).transpose(1, 2)  # (B, h, Lk, dk)
        V = self.W_v(kv_in).view(B, Lk, self.h, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)             # (B, h, Lq, Lk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        out = self.drop(weights) @ V                                       # (B, h, Lq, dk)
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        self.last = {"Q": Q.detach(), "K": K.detach(), "V": V.detach(),
                     "scores": scores.detach(), "weights": weights.detach()}
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.l2(self.drop(F.relu(self.l1(x))))


class PositionalEncoding(nn.Module):
    # kind : "sinus", "appris" ou "aucun"
    def __init__(self, d_model, max_len, kind):
        super().__init__()
        self.kind, self.d = kind, d_model
        if kind == "sinus":
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
            self.register_buffer("pe", pe)
        elif kind == "appris":
            self.emb = nn.Embedding(max_len, d_model)

    def forward(self, L):
        if self.kind == "sinus":
            return self.pe[:L]
        if self.kind == "appris":
            return self.emb(torch.arange(L, device=self.emb.weight.device))
        return torch.zeros(L, self.d)


class EncoderLayer(nn.Module):
    def __init__(self, d, h, d_ff, p):
        super().__init__()
        self.attn = MultiHeadAttention(d, h, p)
        self.ff = FeedForward(d, d_ff, p)
        self.n1, self.n2 = LayerNorm(d), LayerNorm(d)
        self.drop = nn.Dropout(p)

    def forward(self, x, mask):
        x = self.n1(x + self.drop(self.attn(x, x, mask)))
        return self.n2(x + self.drop(self.ff(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d, h, d_ff, p):
        super().__init__()
        self.self_attn = MultiHeadAttention(d, h, p)
        self.cross_attn = MultiHeadAttention(d, h, p)
        self.ff = FeedForward(d, d_ff, p)
        self.n1, self.n2, self.n3 = LayerNorm(d), LayerNorm(d), LayerNorm(d)
        self.drop = nn.Dropout(p)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.n1(x + self.drop(self.self_attn(x, x, tgt_mask)))        # masked self-attention
        x = self.n2(x + self.drop(self.cross_attn(x, memory, src_mask)))  # cross-attention
        return self.n3(x + self.drop(self.ff(x)))


def make_src_mask(src):
    return (src != PAD)[:, None, None, :]


def make_tgt_mask(tgt):
    L = tgt.size(1)
    pad = (tgt != PAD)[:, None, None, :]
    sub = torch.tril(torch.ones(L, L, dtype=torch.bool))[None, None]  # on ne voit pas le futur
    return pad & sub


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=64, n_heads=4, n_enc=2, n_dec=2,
                 d_ff=128, dropout=0.1, pe_type="sinus", max_len=64):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD)
        self.src_pe = PositionalEncoding(d_model, max_len, pe_type)
        self.tgt_pe = PositionalEncoding(d_model, max_len, pe_type)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec)])
        self.out = nn.Linear(d_model, tgt_vocab)
        self.drop = nn.Dropout(dropout)
        self.trace = {}  # sorties intermédiaires gardées pour l'interface
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed(self, ids, emb, pe):
        e = emb(ids) * math.sqrt(self.d_model)
        p = pe(ids.size(1)).unsqueeze(0).expand_as(e)
        return e, p

    def encode(self, src, src_mask):
        e, p = self._embed(src, self.src_emb, self.src_pe)
        x = self.drop(e + p)
        outs = [(e + p).detach()]
        for layer in self.enc_layers:
            x = layer(x, src_mask)
            outs.append(x.detach())
        self.trace.update(src_emb=e.detach(), src_pe=p.detach(), enc_outputs=outs)
        return x

    def decode(self, tgt, memory, src_mask, tgt_mask):
        e, p = self._embed(tgt, self.tgt_emb, self.tgt_pe)
        x = self.drop(e + p)
        outs = [(e + p).detach()]
        for layer in self.dec_layers:
            x = layer(x, memory, src_mask, tgt_mask)
            outs.append(x.detach())
        self.trace.update(tgt_emb=e.detach(), tgt_pe=p.detach(), dec_outputs=outs)
        return self.out(x)

    def forward(self, src, tgt_in):
        sm = make_src_mask(src)
        mem = self.encode(src, sm)
        return self.decode(tgt_in, mem, sm, make_tgt_mask(tgt_in))


@torch.no_grad()
def greedy_translate(model, sv, tv, words, max_len=25):
    # renvoie les ids générés et, pour chaque étape, (id choisi, proba, 3 alternatives)
    model.eval()
    src = torch.tensor([sv.ids(words) + [EOS]])
    sm = make_src_mask(src)
    mem = model.encode(src, sm)
    ys, steps = [SOS], []
    for _ in range(max_len):
        tgt = torch.tensor([ys])
        probs = torch.softmax(model.decode(tgt, mem, sm, make_tgt_mask(tgt))[0, -1], -1)
        top = torch.topk(probs, 4)
        nxt = int(top.indices[0])
        steps.append((nxt, float(probs[nxt]),
                      [(int(i), float(v)) for v, i in zip(top.values[1:], top.indices[1:])]))
        ys.append(nxt)
        if nxt == EOS:
            break
    return ys, steps
