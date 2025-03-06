import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def compute_ce_loss(pred, target, mask=None):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    loss = criterion(pred, target)
    return loss

def align_loss(grounding_feature, qsn, neg_sample=None):
        
    def _info_nce_loss(x, x_pos, x_neg, temperature=0.07):
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
        x_neg = _sample_negatives(grounding_feature, 32)
    else:
        x_neg = neg_sample
    info_nce_loss = _info_nce_loss(qsn, grounding_feature, x_neg)

    return info_nce_loss



LOSS_FNS = {"ce": compute_ce_loss, "align": align_loss}

class BuildLossFunc():
    def __init__(self, cfgs):
        self.loss_fns = cfgs["optim"]["loss"]

    def __call__(self, pred, target):
        loss = {}
        total_loss = 0
        for loss_fn in self.loss_fns.keys():
            if loss_fn == "ce":
                # vq loss
                if "vq_pred" in pred.keys() and "vq" in self.loss_fns["ce"].keys():
                    _loss = LOSS_FNS[loss_fn](pred["vq_pred"], target["vq_target"])
                    total_loss = total_loss + _loss * self.loss_fns[loss_fn]["vq"]
                    loss[f"vq_loss"] = _loss
                # vqa loss
                _loss = LOSS_FNS[loss_fn](pred["vqa_pred"], target["vqa_target"])
                total_loss = total_loss + _loss * self.loss_fns[loss_fn]["vqa"]
                loss[f"vqa_loss"] = _loss
            elif loss_fn == "align":
                # global
                if "g" in self.loss_fns[loss_fn].keys():
                    if "vq_align_pred" in pred.keys():
                        _loss = LOSS_FNS[loss_fn](pred["vq_align_pred"], target["vq_align_gt"])
                    _loss = _loss + LOSS_FNS[loss_fn](pred["vqa_align_pred"], target["vqa_align_gt"])
                    total_loss = total_loss + _loss * self.loss_fns[loss_fn]["g"]
                    loss[f"ag_loss"] = _loss

        loss["total_loss"] = total_loss
        return loss
    