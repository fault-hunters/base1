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
        self.experts = exp_builder(C, **experts)

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

        self.decoder = dec_builder(
            C, C_out, **dec, n_experts=self.n_experts
        )

    def encode(self, img):
        feats = self.style_enc(img)
        feats = self.experts(feats)

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
    
    def defactorize(self, fact_list):
        feats = {}
        for _key in self.fact_blocks:
            _shape = self.feat_shape[_key]
            _cat_dim = -len(_shape)
            _cat_facts = torch.cat([_fact[_key] for _fact in fact_list], dim=_cat_dim)
            _feat = torch.stack([self.recon_blocks[_key][_i](_cat_facts[:, _i])
                                for _i in range(self.n_experts)], dim=1)
            feats[_key] = _feat

        return feats
    # feature map -> vector
    def feat_to_vec(self,feat: torch.Tensor) -> torch.Tensor:
        """
        feat: [B, C, H, W] 또는 [C, H, W]
        return: [C] (한 이미지의 style/content vector)
        """
        if feat.dim() == 3:        # [C,H,W] 들어오면
            feat = feat.unsqueeze(0)  # [1,C,H,W]

        # 보통 B=1 전제
        v = feat.mean(dim=[2, 3])[0]   # [C]
        v = F.normalize(v, dim=-1)     # 코사인용 L2 정규화
        return v
    # vector cosine similarity
    def cosine_sim_01(self,v1: torch.Tensor, v2: torch.Tensor) -> float:
        """
        v1, v2: [C]
        return: 0~1 범위 유사도
        """
        sim = (v1 * v2).sum()      # [-1, 1]
        sim01 = (sim + 1) / 2      # [0, 1]
        return sim01.item()
    '''
    def decode(self, feats):
        out = self.decoder(**feats)
        return out
    '''
    def gen_from_style_char(self, img1, img2):
        B1 = len(img1)
        style_enc_res1 = self.encode(img1)
        style_facts1 = self.factorize(style_enc_res1, 0)
        char_facts1 = self.factorize(style_enc_res1, 1)
        m_style_facts1 = {_k: utils.add_dim_and_reshape(_v, 0, (B1, -1)).mean(1) for _k, _v in style_facts1.items()}
        m_char_facts1 = {_k: utils.add_dim_and_reshape(_v, 0, (B1, -1)).mean(1) for _k, _v in char_facts1.items()}

        B2 = len(img2)
        style_enc_res2 = self.encode(img2)
        style_facts2 = self.factorize(style_enc_res2, 0)
        char_facts2 = self.factorize(style_enc_res2, 1)
        m_style_facts2 = {_k: utils.add_dim_and_reshape(_v, 0, (B2, -1)).mean(1) for _k, _v in style_facts2.items()}
        m_char_facts2 = {_k: utils.add_dim_and_reshape(_v, 0, (B2, -1)).mean(1) for _k, _v in char_facts2.items()}

        # skip + last
        style_feat1   = self.fuser_style(m_style_facts1["last"],   m_style_facts1["skip"])
        style_feat2   = self.fuser_style(m_style_facts2["last"],   m_style_facts2["skip"])

        content_feat1 = self.fuser_content(m_char_facts1["last"],  m_char_facts1["skip"])
        content_feat2 = self.fuser_content(m_char_facts2["last"],  m_char_facts2["skip"])

        # 유사도 계산
        style_feat1 = self.feat_to_vec(style_feat1)
        style_feat2 = self.feat_to_vec(style_feat2)
        content_feat1 = self.feat_to_vec(content_feat1)
        content_feat2 = self.feat_to_vec(content_feat2)

        similarity_style = self.cosine_sim_01(style_feat1, style_feat2)
        similarity_char = self.cosine_sim_01(content_feat1, content_feat2)

        return similarity_style, similarity_char
