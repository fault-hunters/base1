"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .style_encoder import style_enc_builder
from .experts import exp_builder
from .decoder import dec_builder

import utils



class FuseLastSkip(nn.Module):
    def __init__(self, C_last, C_skip, C_out):
        super().__init__()
        self.proj = nn.Conv2d(C_last + C_skip, C_out, 1, 1, 0)

    def forward(self, last, skip):
        # last: [B, C_l, H_l, W_l]
        # skip: [B, C_s, H_s, W_s]
        if skip.shape[2:] != last.shape[2:]:
            skip = F.adaptive_avg_pool2d(skip, last.shape[2:])  # skip 해상도를 last에 맞춤
        x = torch.cat([last, skip], dim=1)  # [B, C_l+C_s, H_l, W_l]
        x = self.proj(x)                    # [B, C_out, H_l, W_l]
        return x


class Generator(nn.Module):
    def __init__(self, C_in, C, C_out, style_enc, experts, emb_num, dec):
        super().__init__()
        self.style_enc = style_enc_builder(
            C_in, C, **style_enc)
        self.experts_s = exp_builder(C, **experts)

        self.contnet_enc = style_enc_builder(
            C_in, C, **style_enc)
        self.experts_c = exp_builder(C, **experts)

        self.n_experts = self.experts.n_experts
        self.feat_shape = {"last": self.experts.out_shape, "skip": self.experts.skip_shape}

        self.fact_blocks = {}
        self.recon_blocks = {}

        self.C_l = self.feat_shape["last"][0]
        self.C_s = self.feat_shape["skip"][0]
        self.fuser_style = FuseLastSkip(self.C_l, self.C_s, C_out=self.C_l)
        self.fuser_content  = FuseLastSkip(self.C_l, self.C_s, C_out=self.C_l)

        self.emb_num = emb_num
        for _key in self.feat_shape:
            _feat_C = self.feat_shape[_key][0]
            self.fact_blocks[_key] = nn.ModuleList([nn.Conv2d(_feat_C, emb_num*_feat_C, 1, 1)
                                                    for _ in range(self.n_experts)])
            self.recon_blocks[_key] = nn.ModuleList([nn.Conv2d(emb_num*_feat_C, _feat_C, 1, 1)
                                                    for _ in range(self.n_experts)])

        self.fact_blocks = nn.ModuleDict(self.fact_blocks)
        self.recon_blocks = nn.ModuleDict(self.recon_blocks)

    def style_encode(self, img):
        feats = self.style_enc(img)
        feats = self.experts_s(feats)

        return feats
    
    def content_encode(self, img):
        feats = self.contnet_enc(img)
        feats = self.experts_c(feats)

        return feats

    def factorize(self, feats, emb_dim=0):
        if self.emb_num is None:
            raise ValueError("embedding blocks are not defined.")

        factors = {}
        for _key, _feat in feats.items():
            _fact = []
            for _i in range(self.n_experts):
                _fact_i = self.fact_blocks[_key][_i](_feat[:, _i])
                _fact_i = utils.add_dim_and_reshape(_fact_i, 1, (self.emb_num, -1))  # (bs*n_s, n_exp, emb_num, *feat_shape)
                _fact.append(_fact_i[:, emb_dim])
            _fact = torch.stack(_fact, dim=1)
            factors[_key] = _fact

        return factors
    
    def feat_to_vec(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)
        v = feat.mean(dim=[2, 3])      # [B,C]
        v = F.normalize(v, dim=1)      # L2 normalize
        return v

    def extract_style_content(self, img: torch.Tensor):
        style_facts = self.style_encode(img)
        char_facts = self.content_encode(img)   # experts까지
        style_facts = self.factorize(style_facts, 0)
        char_facts  = self.factorize(char_facts, 1)

        style_last = style_facts["last"].mean(1)
        style_skip = style_facts["skip"].mean(1)
        char_last  = char_facts["last"].mean(1)
        char_skip  = char_facts["skip"].mean(1)

        style_feat   = self.fuser_style(style_last, style_skip)
        content_feat = self.fuser_content(char_last, char_skip)

        style_vec   = self.feat_to_vec(style_feat)     # [B,C]
        content_vec = self.feat_to_vec(content_feat)   # [B,C]
        return style_vec, content_vec
