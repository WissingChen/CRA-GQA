import torch.nn as nn
import torch
import torch.nn.functional as F
from modules.m_cra.only_trans import clones, PositionwiseFeedForward, LayerNorm, SublayerConnection, MultiHeadedAttention
from modules.m_cra.grounding import GroundingModule
from modules.m_cra.causal_module import VisualIntervention
import math


class RefineMHA(MultiHeadedAttention):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__(h, d_model, dropout)
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.attention = self.refined_attention

        self.attn = None
        self.refined_mask = None
    
    def ori_attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
    
    def refined_attention(self, query, key, value, mask=None, dropout=None):
        # refined mask
        bs, n_head, n_query, d_k = query.size()
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attn = F.softmax(scores, dim=-1)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)\
                .expand(-1, n_head, n_query, -1).reshape(*attn.shape)
            attn = attn * (mask + 1e-10)
            attn = attn / attn.sum(dim=-1, keepdim=True)
        
        qkv = torch.matmul(attn, value)
        return qkv
    
    def causal_attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        neg_scores = torch.matmul(query, -key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            neg_scores = neg_scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        neg_p_attn = F.softmax(neg_scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value) + torch.matmul(neg_p_attn, value), (p_attn, neg_p_attn)

    def forward(self, query, key, value, mask=None, alpha=[0.125, 0.5]):
        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = RefineMHA(num_heads, embed_dim, dropout)

        self.sublayer_ff = SublayerConnection(embed_dim, dropout)
        
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim, dropout) 
        self.sublayer_self = SublayerConnection(embed_dim, dropout)

    def forward(self, x, mask=None, wm_mode=False):
        x = self.sublayer_self(x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer_ff(x, lambda x: self.feed_forward(x))
        return x
    

class Decoder(nn.Module):
    def __init__(self, embed_dim=768, num_layer=2, num_heads=8, ff_dim=768, dropout=0.1, cfgs=None):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.cfgs = cfgs
        self.layers = clones(DecoderLayer(embed_dim, num_heads, ff_dim, dropout), num_layer)
        self.norm = LayerNorm(embed_dim)

        self.attention_video_feature = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
            nn.Softmax(dim=-2)
            )
        
        self.video_proj = nn.Sequential(
            nn.Dropout(dropout),   
            nn.Linear(embed_dim, embed_dim)
            )
        
        #### our proposed module ####
        self.grounding_module = GroundingModule()
        self.do = VisualIntervention(self.cfgs)

    def forward(self, x, qa, self_mask=None, gsub=None, do=False, window_sizes=[1, 3, 5]):
        #####################################
        # encoding for the time relation
        #####################################
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x_bar = self.norm(x)

        #####################################
        # generate the grounding
        #####################################
        time_param = self.grounding_module(x_bar, qa) # cross-model grounding
        
        # attn = self.attention_video_feature(x_bar)
        #####################################
        # NOTE train the recon module 
        # 这里是需要修改的部分，具体的loss输入可以在pipeline/emcl_pipeline.py里修改，loss的计算方式在utils/optim/loss.py
        # 我这里先注释了
        #####################################
        recon_feature = None # self.recon_model(x_bar, qa, self.layers, self.norm, time_param, self.grounding_module)
        # recon_feature["target_video_proj"] = x_bar

        #####################################
        # get the video proj with grounding
        #####################################
        video_proj = torch.sum(x_bar * (time_param["key"]).unsqueeze(dim=-1), dim=1) 
        # video_proj = torch.sum(x_bar * attn, dim=1) 

        #####################################
        # visual causal intervention
        #####################################
        video_proj = self.video_proj(video_proj)
        
        if do:
            x_star = torch.mean(x_bar, dim=1)
            x_star = self.video_proj(x_star)
            video_proj = self.do(x_star, video_proj)

        #####################################
        # align feature
        #####################################
            
        time_param["grounding_feature"] = video_proj
        return video_proj, time_param["key"], time_param
