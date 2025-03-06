"""
qsn_g + vis_token (non-causal) + mask_token -> recon_token

recon_token -> recon time
"""

import torch
from torch import nn
from torch.nn import functional as F


class ReconModule(nn.Module):
    def __init__(self):
        super().__init__()
        # self.recon_loss = nn.MSELoss()
        # self.pos_embedding = nn.Parameter(torch.zeros(1, 32, 768))
        # torch.nn.init.normal_(self.pos_embedding, std=.02)
    
    def _sample_negatives(self, x_pos, k):
        """
        Sample k negative examples for each positive example from x_pos.
        
        Args:
        - x_pos (torch.Tensor): Positive sample tensor of shape [bs, n, d]
        - k (int): Number of negative samples to draw for each positive example
        
        Returns:
        - x_neg (torch.Tensor): Negative samples tensor of shape [bs, k, n, d]
        """
        bs, n, d = x_pos.size()
        x_neg = torch.zeros(bs, k, n, d).to(x_pos.device)
        """
        for i in range(k):
            x_neg[:, i] = torch.roll(x_pos, shifts=i, dims=0)
        """
        for i in range(bs):
            indices = list(range(bs))
            indices.remove(i)
            neg_indices = torch.tensor(indices).to(x_pos.device)
            sampled_indices = neg_indices[torch.randint(0, len(neg_indices), (k,))]
            x_neg[i] = x_pos[sampled_indices]
        return x_neg

    
    def forward(self, x, qa, layers, norm_layer, time_param, gm):
        """
        x    [B, L, D]
        qa   [B, D]
        """
        B, L, D = x.size()
        k = 4
        mask_expanded = time_param["mask"].unsqueeze(dim=-1).unsqueeze(dim=1).expand(B, k, L, D)  # shape: [bs, length, dim]
        # mean_x = torch.mean(x, dim=1, keepdim=True)
        # sample_x_start = x[:, :1]
        # sample_x_end = x[:, -1:]
        # temp = torch.linspace(0, 1, L).to(x.device)
        # pos_embedding = self.pos_embedding.expand(B, -1, -1)
        # temp = temp.view(1, L, 1)  # 形状变为 [1, L, 1]
        # temp = temp.expand(B, L, D)  # 扩展到 [B, L, D]
        # x_replaced = sample_x_start + (sample_x_end - sample_x_start) * temp
        neg_sample = self._sample_negatives(x, k=k)
        x = x.unsqueeze(dim=1).expand(B, k, L, D)
        qa = qa.unsqueeze(dim=1).expand(B, k, D)
        x_replaced = torch.where(mask_expanded == 1, x, neg_sample.squeeze(1))
        # x_replaced = torch.cat([qa.unsqueeze(dim=1), x_replaced], dim=1)
        # for i in range(len(layers)):
            # x_replaced = layers[i](x_replaced)
        # recon_x = norm_layer(x_replaced)[:, 1:]
        x_replaced = x_replaced.view(-1, L, D)
        qa = qa.reshape(-1, D)
        recon_time_param = gm(x_replaced, qa) # cross-model grounding
        
        # recon_video_proj = torch.sum(recon_x * (recon_time_param["key"]).unsqueeze(dim=-1), dim=1) # posthoc_feature + grounding_feature

        return {"video_proj": x_replaced, "time_param": recon_time_param}
    
    def get_recon_loss(self, input, target):
        return self.recon_loss(input, target)
    
    def get_kl_loss(self, input, target):
        return F.kl_div(input.log(), target, reduction='batchmean')
