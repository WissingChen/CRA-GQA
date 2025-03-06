import torch
from torch import nn
from modules.m_emcl.only_trans import clones, PositionwiseFeedForward, LayerNorm, SublayerConnection
import math
from torch.nn import functional as F


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
    
    def forward(self, query, key, value, mask=None):
        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(DecoderLayer, self).__init__()
        # self.attn = MultiHeadedAttention(num_heads, embed_dim, dropout)
        # self.sublayer_attn = SublayerConnection(embed_dim, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.sublayer_ff = SublayerConnection(embed_dim, dropout)

    def forward(self, layer, x, qa, grounding_mask):
        query = torch.cat([qa, x], dim=1)
        query = layer.sublayer_self(query, lambda query: layer.self_attn(query, query, query))
        # x = self.sublayer_attn(x, lambda x: self.attn(x, qa, qa, self_mask))
        qa = query[:, :1]
        x = query[:, 1:]
        x = self.sublayer_ff(x, self.feed_forward) + layer.sublayer_ff(x, layer.feed_forward).detach()
        return x, qa


class WorldDecoder(nn.Module):
    def __init__(self, embed_dim=768, num_layer=2, num_heads=8, ff_dim=768, dropout=0.1):
        super(WorldDecoder, self).__init__()
        self.layers = clones(DecoderLayer(embed_dim, num_heads, ff_dim, dropout), num_layer)
        self.norm = LayerNorm(embed_dim)

    def forward(self, layers, x, qa, grounding_mask):
        for i in range(len(self.layers)):
            x, _ = self.layers[i](layers[i], x, qa, grounding_mask)
        x = self.norm(x)
        return x

    
    def _sample_negatives(self, qa_pos, k):
        bs, _, n = qa_pos.size()
        qa_pos = qa_pos.squeeze(1)

        qa_neg = torch.zeros(bs, k, n).to(qa_pos.device)
        for i in range(bs):
            indices = list(range(bs))
            indices.remove(i)
            neg_indices = torch.tensor(indices).to(qa_pos.device)
            sampled_indices = neg_indices[torch.randint(0, len(neg_indices), (k,))]
            qa_neg[i] = qa_pos[sampled_indices]
        return qa_neg

    
    def neg_sample(self, layers, x, qa_pos, keyframe_probs, grounding_mask, k=16):
        bs, len_video, n = x.size()
        neg_sample = torch.zeros(bs, k, n).to(qa_pos.device)
        qa_neg = self._sample_negatives(qa_pos, k)
        for _k in range(k):
            temp_x = x
            self_mask = None
            _qa_neg = qa_neg[:, _k:_k+1]
            for i in range(len(self.layers)):
                temp_x, _ = self.layers[i](layers[i], temp_x, _qa_neg, self_mask)
            temp_x = self.norm(temp_x)

            temp_x = x * ( ~grounding_mask.unsqueeze(-1)) + temp_x * grounding_mask.unsqueeze(-1)

            temp_x = torch.sum(temp_x * keyframe_probs.unsqueeze(-1), dim=1) # [bs, n]
            neg_sample[:, _k] = temp_x
        return neg_sample


class VisualWorldModel(nn.Module):
    def __init__(self, d_model=768, num_vec=512):
        super(VisualWorldModel, self).__init__()
        """Visual World Model
        1. neg sample: gen the neg sample by the corrected visual feature and incorrected qa
        2. x_hat for intervention: 
        """
        self.d_model = d_model
        self.world_model = WorldDecoder()
        self.recon_loss = nn.MSELoss()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        # self.world_vec = nn.Parameter(torch.zeros(num_vec, self.embed_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)

    
    def forward(self, layers, ori_x, x, qa, grounding_mask, keyframe_probs):
        """
        x: visual feature [batch_size, len_video, d_model]
        qa: glabol qa feature [batch_size, 1, d_model]
        grounding_mask: the grounding region is True, otherwise is False [batch_size, len_video]
        keyframe_probs: [batch_size, len_video]

        1. mask the grounding region
        2. recon the feature via qa feature and unmask feature
        3. get the neg sample via incorrect qa
        """
        bs, len_video, d_model = ori_x.size()
        qa = qa.reshape([bs, 1, d_model])
        mask_token = self.mask_token.repeat([bs, len_video, 1])
        masked_token = x * ( ~grounding_mask.unsqueeze(-1)) + mask_token * grounding_mask.unsqueeze(-1)

        recon_feature = self.world_model(layers, masked_token, qa, grounding_mask)
        neg_sample = self.world_model.neg_sample(layers, masked_token, qa, keyframe_probs, grounding_mask, k=bs//8)

        recon_loss = self.recon_loss(x*grounding_mask.unsqueeze(-1), recon_feature*grounding_mask.unsqueeze(-1))
        # recon_loss = self.recon_loss(x, recon_feature)
        world_feature = {"recon_feature": recon_feature, "neg_sample": neg_sample, "recon_loss": recon_loss}
        return world_feature