import os
import torch
from torch import nn
from torch.nn import functional as F
import math
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class AdaptiveGaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, initial_sigma=1.0, max_sigma=5.0):
        super(AdaptiveGaussianFilter, self).__init__()
        self.kernel_size = kernel_size  # Size of the Gaussian kernel
        self.padding = kernel_size // 2  # Padding to keep the output size same as input
        
        # Learnable or fixed parameter: sigma (initially set to the given value)
        self.sigma = nn.Parameter(torch.tensor(initial_sigma, dtype=torch.float32))
        self.max_sigma = max_sigma
        """
        self.sigma = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        """

    def forward(self, weights):
        """
        weights: input of shape (batch_size, num_frames, 1), where num_frames is the number of time steps (e.g., 32).
        """
        # Generate the Gaussian kernel dynamically based on sigma
        kernel = self.create_gaussian_kernel(self.kernel_size, self.sigma, device=weights.device)
        
        # Apply Gaussian smoothing using 1D convolution (along the frame dimension)
        weights = weights.permute(0, 2, 1)
        # smoothed_weights = torch.zeros_like(weights).cuda()
        # for i in range(kernel.size(0)):
        smoothed_weights = F.conv1d(weights, kernel, padding=self.padding)
        
        smoothed_weights = smoothed_weights.permute(0, 2, 1)
        # After smoothing, we need to re-normalize the weights to ensure they sum to 1 (like Softmax)
        smoothed_weights = F.softmax(smoothed_weights, dim=1)
        return smoothed_weights  # Return to original shape

    def create_gaussian_kernel(self, kernel_size, sigma, device): # single sigma
        # Create a range of values from -(kernel_size//2) to +(kernel_size//2)
        x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        
        # Compute the Gaussian kernel (1D)
        gaussian_kernel = torch.exp(-0.5 * (x / sigma).pow(2))
        
        # Normalize the kernel so that its sum equals 1
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        
        # Reshape to (out_channels, in_channels, kernel_size) for Conv1d
        return gaussian_kernel.view(1, 1, -1)



class GroundingModule(nn.Module):
    def __init__(self, d_model=1536, dropout=0.3):
        super().__init__()
        self.qa_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU()
            )
    
        self.v_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU()
            )
        
        self.grounding = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Softmax(dim=-2)
                )
    
        self.gs_filter = AdaptiveGaussianFilter()
    
    def _get_pos_embedding(self, max_len=32, d_model=768):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe  # [1, 32, 768]

    
    def gen_grounding(self, keyframe_prob, window_sizes=[1, 3, 5]):
        bs, length = keyframe_prob.size()[:2]
        max_indices = self.find_best_interval_v2(keyframe_prob)
        start_indices = max_indices[:, 0].unsqueeze(1).expand(-1, length)
        end_indices = max_indices[:, 1].unsqueeze(1).expand(-1, length)
        range_tensor = torch.arange(length, device=keyframe_prob.device).expand(bs, length)
        keyframe_mask = (range_tensor >= start_indices) & (range_tensor <= end_indices)
        start_time = max_indices[:, 0] / 31.
        end_time = max_indices[:, 1] / 31.
        time_param = {"key": keyframe_prob, "max_indices": max_indices, "start": start_time, "end": end_time, "mask": keyframe_mask}
        return time_param
    
    def find_best_interval_v1(self, keyframe_prob, window_sizes=[1, 3, 5]):
        bs, length = keyframe_prob.size()[:2]
        max_probs = torch.zeros(bs).to(keyframe_prob.device)
        max_indices = torch.zeros(bs, 2, dtype=torch.short).to(keyframe_prob.device)
        overall_max, _ = keyframe_prob.max(dim=1, keepdim=True)
        for window_size in window_sizes:
            for i in range(length - window_size + 1):
                window_probs = keyframe_prob[:, i:i + window_size].sum(dim=1)
                
                window_max, _ = keyframe_prob[:, i:i + window_size].max(dim=1)
                
                valid_mask = (window_max == overall_max.squeeze())
                max_mask = (window_probs > max_probs) & valid_mask
                
                max_probs[max_mask] = window_probs[max_mask]
                max_indices[max_mask, 0] = i
                max_indices[max_mask, 1] = i + window_size - 1
        
        return max_indices

    def find_best_interval_v2(self, keyframe_prob, window_sizes=[1, 3, 5]):
        bs, length = keyframe_prob.shape
        max_interval_size = length // 2
        max_indices = torch.zeros((bs, 2), dtype=torch.long, device=keyframe_prob.device)

        # Initialize a tensor to store the maximum scores and their corresponding indices
        best_scores = torch.full((bs,), float('-inf'), dtype=torch.float, device=keyframe_prob.device)
        
        for window_size in window_sizes:
            if window_size > max_interval_size:
                continue

            # Calculate sliding window sums for all batches simultaneously using 1D convolution
            sliding_sums = F.conv1d(
                keyframe_prob.unsqueeze(1), 
                weight=torch.ones((1, 1, window_size), device=keyframe_prob.device), 
                padding=0, 
                stride=1
            ).squeeze(1)

            # Create a mask to ensure the max value in the distribution is within the interval
            max_values = keyframe_prob.max(dim=1, keepdim=True).values
            max_positions = keyframe_prob.argmax(dim=1, keepdim=True)

            for start in range(length - window_size + 1):
                end = start + window_size
                contains_max = (max_positions >= start) & (max_positions < end)
                
                window_scores = sliding_sums[:, start]
                window_scores[~contains_max.squeeze()] = float('-inf')
                
                # Update best scores and indices
                better_scores = window_scores > best_scores
                best_scores = torch.where(better_scores, window_scores, best_scores)
                max_indices[better_scores] = torch.tensor([start, end], device=keyframe_prob.device)

        return max_indices
    
    def _sample_negatives(self, x_pos, k):
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
    
    
    def forward(self, v, qa):
        v = self.v_proj(v)
        qa = self.qa_proj(qa)
        gate = (torch.matmul(v, qa.unsqueeze(dim=-1))).tanh()#  # bs, length, 1
        keyframe_prob = self.grounding(v*gate) # bs, length, 1
        keyframe_prob_gs = self.gs_filter(keyframe_prob)  # [bs, length]
        keyframe_prob_gs = keyframe_prob_gs.squeeze(dim=-1)
        time_param = self.gen_grounding(keyframe_prob_gs)
        time_param["neg_key"] = self._sample_negatives(keyframe_prob_gs, v.size(0)//8)
        time_param["ori_key"] = keyframe_prob.squeeze(dim=-1)
        return time_param


class FrontDoorIntervention(nn.Module):
    def __init__(self):
        super().__init__()
        embedding_size = 1536
        self.prob_m = nn.Linear(embedding_size, 768)
        self.prob_x_hat = nn.Linear(768, 768)
        self.prob_x = nn.Linear(embedding_size, 768)

        self.prob_x_m = nn.Linear(768, embedding_size)
        self.prob_x_hat_x = nn.Linear(768, embedding_size)

        self.fc = nn.Linear(embedding_size*2, embedding_size)

        # self.norm = LayerNorm(768)

    def forward(self, x, m, x_hat):
        # P(M|X)P(Y|X_hat, M)P(X_hat)
        prob_m = self.prob_m(m)
        prob_x_hat = self.prob_x_hat(x_hat)
        prob_x = self.prob_x(x)

        # P(M|X)
        prob_x_m = torch.matmul(prob_m, prob_x.transpose(-1, -2)) / math.sqrt(m.size(-1))
        prob_x_m = F.softmax(prob_x_m, dim=-1)
        prob_x_m = self.prob_x_m(torch.matmul(prob_x_m, prob_x))

        # P(Y|X_hat, M)P(X_hat)
        prob_x_hat_x = torch.matmul(prob_x, prob_x_hat.transpose(-1, -2)) / math.sqrt(m.size(-1))
        prob_x_hat_x = F.softmax(prob_x_hat_x, dim=-1)
        prob_x_hat_x = self.prob_x_hat_x(torch.matmul(prob_x_hat_x, prob_x_hat))
        
        out = self.fc(torch.cat([prob_x_m, prob_x_hat_x], dim=-1))

        # out = self.norm(out + m)
        return out


class BackDoorIntervention(nn.Module):
    def __init__(self):
        super().__init__()

        self.q = nn.Linear(768*2, 768)
        self.k = nn.Linear(768, 768)
        self.v = nn.Linear(768, 768*2)

        # self.norm = LayerNorm(768)

    def forward(self, x, confounder):
        # confounder = self.confounder.to(x.device)
        
        q = self.q(x)
        k = self.k(confounder)
        v = self.v(confounder)

        qk = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        score = F.softmax(qk, dim=-1)
        out = torch.matmul(score, v)

        # out = self.norm(out + x)
        return out


class LinguisticInterventionGraph(nn.Module):
    def __init__(self, causal_feature_path):
        super(LinguisticInterventionGraph, self).__init__()
        qa_graphs = torch.load(os.path.join(causal_feature_path, "qa_graphs.npy"))["k_center"]
        node = qa_graphs.x
        self.edge = qa_graphs.edge_index
        self.node = nn.Parameter(node.cuda().to(torch.float32).requires_grad_())

        self.norm = nn.LayerNorm(768*2)

        self.gcn = GCNConv(768, 768//8)

        # self.fc = nn.Linear(768, 768 // 8)
        # self.fc2 = nn.Linear(768, 768)
        
        self.do = BackDoorIntervention()

    def forward(self, x):
        edge = self.edge.to(x.device)
        node = self.node.to(x.device)
        confounder = self.gcn(node, edge)
        # confounder = self.norm(confounder)
        # confounder = self.fc(confounder).reshape(-1, 768)
        confounder = confounder.reshape(-1, 768)
        return self.do(x, confounder)


class VisualIntervention(nn.Module):
    def __init__(self, causal_feature_path):
        super(VisualIntervention, self).__init__()
        visual_clusters = torch.load(os.path.join(causal_feature_path, "visual_clusters.npy"))["k_center"]
        self.visual_clusters = nn.Parameter(visual_clusters.cuda().to(torch.float32).requires_grad_())
        self.do = FrontDoorIntervention()

    def forward(self, x, m):
        visual_clusters = self.visual_clusters.to(x.device)
        return self.do(x, m, visual_clusters)


class Intervention(nn.Module):
    def __init__(self):
        super().__init__()
        # self.cfgs = cfgs
        self._vis_do = VisualIntervention("../../../data/nextgqa/causal_feature")
        self._lin_do = LinguisticInterventionGraph("../../../data/nextgqa/causal_feature")

        self.vis_norm = nn.LayerNorm(1536)
        self.lin_norm = nn.LayerNorm(1536)

    def lin_do(self, x):
        x = self._lin_do(x) + x
        x = self.lin_norm(x)
        return x
    
    def vis_do(self, x, m):
        vis_m = self._vis_do(x, m)
        # m = vis_m + m
         #m = self.vis_norm(m)
        return vis_m

    def vocab_lin_do(self, x):
        x = self.norm_vocab(self.vocab_do(x) + x)
        return x


def align_loss(grounding_feature, qsn, qgt=None, neg_sample=None):
        
    def _info_nce_loss(x, x_pos, x_neg, qgt, temperature=0.07):
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
        # pos_sim = torch.bmm(x.unsqueeze(1), x_pos.transpose(1, 2)).squeeze(1) / temperature
        
        # Calculate negative similarities
        neg_sim = torch.bmm(x.unsqueeze(1), x_neg.transpose(1, 2)).squeeze(1) / temperature
        
        # Concatenate positive and negative similarities
        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        
        # Create labels for the positive samples
        labels = torch.zeros(x.size(0), dtype=torch.long).to(x.device)
        # labels = torch.tensor(qgt, dtype=torch.long).to(x.device)
        
        # Compute the cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
        
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
    
    if qgt is not None:
        _grounding_feature = grounding_feature[[torch.arange(qsn.size(0)), qgt]]
        qsn = qsn[[torch.arange(qsn.size(0)), qgt]]
    x_neg = _sample_negatives(_grounding_feature, _grounding_feature.size(0)//2)
    info_nce_loss = _info_nce_loss(qsn, _grounding_feature, x_neg, qgt)

    return info_nce_loss