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
    def __init__(self, d_model=768, dropout=0.3):
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
    
    def gen_grounding(self, keyframe_prob, window_sizes=[1, 3, 5]):
        bs, length = keyframe_prob.size()[:2]
        max_indices = self.find_best_interval(keyframe_prob)
        start_indices = max_indices[:, 0].unsqueeze(1).expand(-1, length)
        end_indices = max_indices[:, 1].unsqueeze(1).expand(-1, length)
        range_tensor = torch.arange(length, device=keyframe_prob.device).expand(bs, length)
        keyframe_mask = (range_tensor >= start_indices) & (range_tensor <= end_indices)
        start_time = max_indices[:, 0] / 31.
        end_time = max_indices[:, 1] / 31.
        time_param = {"key": keyframe_prob, "max_indices": max_indices, "start": start_time, "end": end_time, "mask": keyframe_mask}
        return time_param

    def find_best_interval(self, keyframe_prob, window_sizes=[1, 3, 5]):
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
    
    def forward(self, v, qa):
        v = self.v_proj(v)
        qa = self.qa_proj(qa)
        gate = (torch.matmul(v, qa.unsqueeze(dim=-1))).tanh()#  # bs, length, 1
        keyframe_prob = self.grounding(v*gate) # bs, length, 1
        keyframe_prob_gs = self.gs_filter(keyframe_prob)  # [bs, length]
        keyframe_prob_gs = keyframe_prob_gs.squeeze(dim=-1)
        time_param = self.gen_grounding(keyframe_prob_gs)
        time_param["ori_key"] = keyframe_prob.squeeze(dim=-1)
        return time_param

