import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch
import math
from modules.m_cra.only_trans import LanModel, MultiHeadedAttention
from modules.m_tempclip.transformer import VideoEmbedding, LayerNorm
from modules.m_cra.only_trans import Decoder as BaselineDecoder
from modules.m_cra.refine import Decoder as RefineDecoder
from modules.m_cra.causal_module import LinguisticInterventionGraph

class CRA(nn.Module):
    def __init__(self, cfgs, tokenizer) -> None:
        super(CRA, self).__init__()
        self.cfgs = cfgs
        self.tokenizer = tokenizer
        self.baseline = cfgs["model"]["baseline"]
        self.qmax_words = cfgs["dataset"]["qmax_words"]
        self.max_feats = cfgs["dataset"]["max_feats"]
        self.n_negs = cfgs["model"]["n_negs"]

        d_model = cfgs["model"]["d_model"]
        # answer modules
        self.lang_model = LanModel(tokenizer, lan=cfgs["model"]["lan"], out_dim=d_model)
        self.video_model = VideoEmbedding(cfgs)
        self.do = LinguisticInterventionGraph(cfgs)

        if cfgs["model"]["baseline"] == "baseline":
            self.decoder = BaselineDecoder()
            self.forward = self.baseline_forward
        
        elif cfgs["model"]["baseline"] == "refine":
            self.decoder = RefineDecoder(num_layer=cfgs["model"]["num_layers"], cfgs=cfgs)
            self.forward = self.refine_forward

    def _check_device(self, module, x):
        for name, param in module.named_parameters():
            if param.device != x.device:
                print(f"Parameter {name} is on device {param.device}, expected {x.device}")

    def refine_forward(self, video, video_mask, answer=None, answer_id=None, mode="VQA", gsub=None):
        answer_g, answer_w = self.lang_model(answer.to("cuda")) # [bs, mc, D] [bs, mc, l, D]

        #####################################
        # linguistics causal intervention
        #####################################
        
        # self._check_device(self.video_model, video)
        video_f = self.video_model(video)
        qsn_gt = None
        if answer_id is not None and mode == "VQ":
            qsn_g = answer_g[torch.arange(answer_g.shape[0]), answer_id.squeeze(), :]
            qsn_gt = qsn_g
        elif answer_id is not None and mode == "VQA":
            qsn_g = answer_g.max(dim=1)[0]
            qsn_gt = answer_g[torch.arange(answer_g.shape[0]), answer_id.squeeze(), :]
        else:
            qsn_g = answer_g.max(dim=1)[0]

        # if self.cfgs["model"]["do"]:
            # B, M, D = answer_g.size()
            # answer_g = answer_g.view(B*M, D)
            # qsn_g = self.do(qsn_g)  
            # answer_g = answer_g.view(B, M, D)

        video_proj, video_attn, time_param = self.decoder(video_f, qsn_g, video_mask, gsub=gsub, do=self.cfgs["model"]["do"], window_sizes=self.cfgs["model"]["window_sizes"])
        
        answer_proj = answer_g
        
        outs = {"fatt": video_attn, "time_param": time_param, 
                "align_gt": qsn_gt,
                }
        return video_proj, answer_proj, outs

    def baseline_forward(self, video, video_mask, answer=None, answer_id=None, mode="VQA", gsub=None):
        answer_g, answer_w = self.lang_model(answer.cuda()) # [bs, mc, 768], [bs, mc, n, 768]
        video_f = self.video_model(video) # [bs, l, 768] 
        if answer_id is not None:
            qsn_g = answer_g[torch.arange(answer_g.shape[0]), answer_id.squeeze(), :]
        else:
            qsn_g = answer_g.max(dim=1)[0]

        video_proj, video_attn = self.decoder(video_f, answer_w, video_mask)
        answer_proj = answer_g

        outs = {"fatt": video_attn}
        return video_proj, answer_proj, outs

    def align_loss(self, qsn, grounding_feature, neg_sample=None):
        
        def _sample_negatives(x_pos, k):
            """
            Sample k negative examples for each positive example from x_pos.
            
            Args:
            - x_pos (torch.Tensor): Positive sample tensor of shape [bs, n]
            - k (int): Number of negative samples to draw for each positive example
            
            Returns:
            - x_neg (torch.Tensor): Negative samples tensor of shape [bs, k, n]
            """
            bs, n = x_pos.size()
            x_neg = torch.zeros(bs, k, n).to(x_pos.device)
            for i in range(bs):
                indices = list(range(bs))
                indices.remove(i)
                neg_indices = torch.tensor(indices).to(x_pos.device)
                sampled_indices = neg_indices[torch.randint(0, len(neg_indices), (k,))]
                x_neg[i] = x_pos[sampled_indices]
            return x_neg
        
        if neg_sample is None:
            x_neg = _sample_negatives(grounding_feature, self.cfgs["dataset"]["batch_size"]//2)
        else:
            x_neg = neg_sample
        info_nce_loss = self._info_nce_loss(qsn, grounding_feature, x_neg)


        return info_nce_loss

    def _info_nce_loss(self, x, x_pos, x_neg, temperature=0.07):
        """
        Compute the InfoNCE loss given query x, positive sample x_pos, and negative samples x_neg.
        
        Args:
        - x (torch.Tensor): Query tensor of shape [bs, n]
        - x_pos (torch.Tensor): Positive sample tensor of shape [bs, n]
        - x_neg (torch.Tensor): Negative samples tensor of shape [bs, k, n]
        - temperature (float): Temperature scaling factor
        
        Returns:
        - loss (torch.Tensor): InfoNCE loss
        """
        # Normalize the input vectors
        x = F.normalize(x, dim=-1)
        x_pos = F.normalize(x_pos, dim=-1)
        x_neg = F.normalize(x_neg, dim=-1)
        
        # Calculate positive similarities
        pos_sim = torch.sum(x * x_pos, dim=-1, keepdim=True) / temperature
        
        # Calculate negative similarities
        neg_sim = torch.bmm(x.unsqueeze(1), x_neg.transpose(1, 2)).squeeze(1) / temperature
        
        # Concatenate positive and negative similarities
        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        
        # Create labels for the positive samples
        labels = torch.zeros(x.size(0), dtype=torch.long).to(x.device)
        
        # Compute the cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss