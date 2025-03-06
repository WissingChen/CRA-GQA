import torch
from torch import nn
from torch.nn import functional as F
import math


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

    def create_gaussian_kernel_v2(self, kernel_size, gate, device):
        sigma = self.sigma(gate.squeeze(dim=-1)).unsqueeze(dim=-1) * self.max_sigma # [bs, 32, 1] -> [bs, 1, 1]
        # Create a range of values from -(kernel_size//2) to +(kernel_size//2)
        x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        x = x.view(1, 1, kernel_size)

        # Compute the Gaussian kernel (1D)
        gaussian_kernels = torch.exp(-0.5 * (x / sigma).pow(2))
        
        # Normalize the kernels so that the sum over kernel_size equals 1
        gaussian_kernels = gaussian_kernels / gaussian_kernels.sum(dim=-1, keepdim=True)
        
        return gaussian_kernels  # shape: (64, 1, kernel_size)


class GroundingModule(nn.Module):
    def __init__(self, d_model=768, dropout=0.3):
        super().__init__()
        # self.pred_duration = nn.Sequential(
        #         nn.Dropout(dropout),
        #         nn.Linear(d_model, 1),
        #         nn.Sigmoid()
        #         )
        # self.pos_embedding = nn.Embedding(32, d_model)
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

        self.time_estimate = nn.Sequential(
                nn.BatchNorm1d(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.BatchNorm1d(d_model // 2),
                nn.Linear(d_model // 2, 3),
                nn.ReLU()
                )
    
        self.gs_filter = AdaptiveGaussianFilter()
        # self.pos_embedding = self._get_pos_embedding().cuda()  # nn.Parameter(torch.rand([32, 768]))
    
    def _get_pos_embedding(self, max_len=32, d_model=768):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe  # [1, 32, 768]
    
    def gen_grounding_v2(self, keyframe_prob, v, alpha=20):
        bs, length = v.size()[:2]
        activated_v = torch.sum(keyframe_prob.unsqueeze(dim=-1)*v, dim=1) # [bs, n]
        time_output = self.time_estimate(activated_v)
        keyframe = time_output[:, 0]
        start = time_output[:, 1]
        end = time_output[:, 2]
        start_time = (keyframe-start).sigmoid()
        end_time = (keyframe+end).sigmoid()
        # 生成mask
        positions = torch.linspace(0, 1, length, device=v.device)# .expand(time_output.size(0), length)  # Shape: (k,)
        
        # Compute the sigmoid-based weights for start and end boundaries
        start_weights = torch.sigmoid(alpha * (positions - start_time.unsqueeze(1)))  # Shape: (batch_size, k)
        end_weights = torch.sigmoid(alpha * (end_time.unsqueeze(1) - positions))      # Shape: (batch_size, k)
        
        # The final weights are the product of the start and end masks
        keyframe_mask = start_weights * end_weights  # Shape: (batch_size, k)
        pos_embedding = keyframe_mask.unsqueeze(dim=-1) * self.pos_embedding
        # range_tensor = torch.arange(length, device=time_output.device).expand(time_output.size(0), length)
        # keyframe_mask = (range_tensor >= start_time.unsqueeze(dim=-1)*(length-1)) & (range_tensor <= end_time.unsqueeze(dim=-1)*(length-1))
        time_param = {"key": keyframe_prob, "start": start_time, "end": end_time, "mask": keyframe_mask, "pos_embed": pos_embedding}
        return time_param
    
    def gen_grounding(self, keyframe_prob, window_sizes=[1, 3, 5]):
        bs, length = keyframe_prob.size()[:2]
        max_indices = self.find_best_interval_v2(keyframe_prob)
        # 将index_pairs扩展为同样的形状，方便广播比较
        start_indices = max_indices[:, 0].unsqueeze(1).expand(-1, length)
        end_indices = max_indices[:, 1].unsqueeze(1).expand(-1, length)
        # 生成mask
        range_tensor = torch.arange(length, device=keyframe_prob.device).expand(bs, length)
        keyframe_mask = (range_tensor >= start_indices) & (range_tensor <= end_indices)
        start_time = max_indices[:, 0] / 31.
        end_time = max_indices[:, 1] / 31.
        time_param = {"key": keyframe_prob, "max_indices": max_indices, "start": start_time, "end": end_time, "mask": keyframe_mask}
        return time_param
    
    def find_best_interval_v1(self, keyframe_prob, window_sizes=[1, 3, 5]):
        bs, length = keyframe_prob.size()[:2]
        # 初始化结果变量
        max_probs = torch.zeros(bs).to(keyframe_prob.device)
        max_indices = torch.zeros(bs, 2, dtype=torch.short).to(keyframe_prob.device)
        overall_max, _ = keyframe_prob.max(dim=1, keepdim=True)
        # 遍历所有窗口大小
        for window_size in window_sizes:
            for i in range(length - window_size + 1):
                window_probs = keyframe_prob[:, i:i + window_size].sum(dim=1)
                
                # 计算每个窗口内最大值的位置
                window_max, _ = keyframe_prob[:, i:i + window_size].max(dim=1)
                
                # 确保窗口包含最大值
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
    
    def time_penalty(self, time_param, lambda_1=1, lambda_2=1):
            """
            时间上的惩罚主要有两项：
            1. 约束关键帧概率集中在grounding内, 区间大小固定为窗的大小, 区间内的总概率大于区间外
            2. 由于约束1的作用可能会使grounding变得过大, 因此需要进一步约束grounding可以在尽可能小的范围内有更大的概率分布.
            所以, 可以通过区间内的方差（尽可能小）和均值（尽可能大）衡量他的集中程度以及效率
            """
            t_start = time_param["start"]
            t_end = time_param["end"]
            probs = time_param["key"]
            # duration = time_param["duration"]
            frame_count = probs.size(1)
            ######################
            # penalty 1
            ######################
            # 预先计算帧的位置
            frame_positions = torch.linspace(0, 1, steps=frame_count).to(t_start.device)
            # 将t_start和t_end扩展到每帧的维度
            t_start = t_start.unsqueeze(1).expand(-1, frame_count)
            t_end = t_end.unsqueeze(1).expand(-1, frame_count)
            
            # 计算每一帧的权重
            
            # 起始之前和终止之后的帧的权重为1，其他为0
            conf_weights = ((frame_positions < t_start) | (frame_positions > t_end)).float()
            
            # 计算关键帧概率的惩罚项，目标是令G以外的概率都为零
            keyframe_neg = (probs * conf_weights).sum(dim=1)
            keyframe_pos = (probs * (1 - conf_weights)).sum(dim=1)
            keyframe_penalty_mean = (keyframe_neg - keyframe_pos).mean()
            ######################
            # penalty 2
            ######################
            # weights = ((frame_positions >= t_start) & (frame_positions <= t_end)).float()
            # g_count = weights.sum(dim=-1)
            # g_mean = (probs * weights).sum(dim=1) / (g_count + 1e-10)
            # g_var = ((probs - g_mean.unsqueeze(-1))**2 * weights).sum(dim=1) /  (g_count + 1e-10)
            # g_mean = g_mean.mean()
            # g_var = g_var.mean()
            time_penalty = keyframe_penalty_mean # + lambda_1 * g_var.mean() - lambda_2 * g_mean.mean()
            time_param["time_penalty"] = time_penalty
            return time_param
    
    def forward(self, v, qa):
        v = self.v_proj(v)
        qa = self.qa_proj(qa)
        gate = (torch.matmul(v, qa.unsqueeze(dim=-1))).tanh()#  # bs, length, 1
        keyframe_prob = self.grounding(v*gate) # bs, length, 1
        keyframe_prob_gs = self.gs_filter(keyframe_prob)  # [bs, length]
        keyframe_prob_gs = keyframe_prob_gs.squeeze(dim=-1)
        time_param = self.gen_grounding(keyframe_prob_gs)
        # time_param["neg_key"] = self._sample_negatives(keyframe_prob_gs, v.size(0)//8)
        time_param["ori_key"] = keyframe_prob.squeeze(dim=-1)
        return time_param

    def forward_v2(self, v, qa):
        v = self.v_proj(v)
        qa = self.qa_proj(qa)
        keyframe_prob = (torch.matmul(v, qa.unsqueeze(dim=-1))).softmax(dim=1) #  # bs, length, 1
        keyframe_prob = self.grounding(v*gate) # bs, length, 1
        keyframe_prob = self.gs_filter(keyframe_prob.squeeze(dim=-1))  # [bs, length]
        time_param = self.gen_grounding_v2(keyframe_prob, v)
        time_param["neg_key"] = self._sample_negatives(time_param["mask"], v.size(0)//8)
        return time_param
