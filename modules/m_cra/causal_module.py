import os
import torch
from torch import nn
from torch.nn import functional as F
import math
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class FrontDoorIntervention(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        
        d_model = cfgs["model"]["d_model"]
        d_feat = cfgs["model"]["feature_dim"]
        self.prob_m = nn.Linear(d_model, d_model)
        self.prob_x_hat = nn.Linear(d_feat, d_model)
        self.prob_x = nn.Linear(d_model, d_model)

        self.prob_x_m = nn.Linear(d_model, d_model)
        self.prob_x_hat_x = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model*2, d_model)

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

        self.q = nn.Linear(768, 768)
        self.k = nn.Linear(768, 768)
        self.v = nn.Linear(768, 768)

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



class LinguisticInterventionVocab(nn.Module):
    def __init__(self, causal_feature_path):
        super(LinguisticInterventionVocab, self).__init__()
        qa_vocab = torch.load(os.path.join(causal_feature_path, "qa_noun_vocab.npy"))["k_center"]
        qa_vocab = torch.from_numpy(qa_vocab)
        self.qa_vocab = nn.Parameter(qa_vocab.cuda().to(torch.float32).requires_grad_())
        self.do = BackDoorIntervention()

    def forward(self, x):
        confounder = self.qa_vocab.to(x.device)
        return self.do(x, confounder)


class LinguisticInterventionGraphFID(nn.Module):
    def __init__(self, causal_feature_path):
        super(LinguisticInterventionGraphFID, self).__init__()
        qa_graphs = torch.load(os.path.join(causal_feature_path, "qa_graphs.npy"))["k_center"]
        node = qa_graphs.x
        self.edge = qa_graphs.edge_index
        self.node = nn.Parameter(node.cuda().to(torch.float32).requires_grad_())

        self.norm = nn.LayerNorm(768)

        self.gcn = GCNConv(768, 768//8)

        # self.fc = nn.Linear(768, 768 // 8)
        # self.fc2 = nn.Linear(768, 768)
        
        self.do = FrontDoorIntervention()# BackDoorIntervention()

    def forward(self, x, m):
        edge = self.edge.to(x.device)
        node = self.node.to(x.device)
        x_hat = self.gcn(node, edge)
        # confounder = self.norm(confounder)
        # confounder = self.fc(confounder).reshape(-1, 768)
        x_hat = x_hat.reshape(-1, 768)
        return self.do(x, m, x_hat)



class LinguisticInterventionGraph(nn.Module):
    def __init__(self, cfgs):
        super(LinguisticInterventionGraph, self).__init__()
        qa_graphs = torch.load(os.path.join(cfgs["dataset"]["causal_feature_path"], "qa_graphs.npy"))["k_center"]
        node = qa_graphs.x
        self.edge = qa_graphs.edge_index
        self.node = nn.Parameter(node.cuda().to(torch.float32).requires_grad_())

        self.norm = nn.LayerNorm(768)

        self.gcn = GCNConv(768, 768//8)
        
        self.do = BackDoorIntervention()

    def forward(self, x):
        edge = self.edge.to(x.device)
        node = self.node.to(x.device)
        confounder = self.gcn(node, edge)
        confounder = confounder.reshape(-1, 768)
        x = self.do(x, confounder) + x
        x = self.norm(x)
        return x


class VisualIntervention(nn.Module):
    def __init__(self, cfgs):
        super(VisualIntervention, self).__init__()
        visual_clusters = torch.load(os.path.join(cfgs["dataset"]["causal_feature_path"], "visual_clusters.npy"))["k_center"]
        self.visual_clusters = nn.Parameter(visual_clusters.cuda().to(torch.float32).requires_grad_())
        self.do = FrontDoorIntervention(cfgs)
        self.norm = nn.LayerNorm(cfgs["model"]["d_model"])

    def forward(self, x, m):
        visual_clusters = self.visual_clusters.to(x.device)
        m = self.do(x, m, visual_clusters) + m
        m = self.norm(m)
        return m


class Intervention(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self._vis_do = VisualIntervention(cfgs)
        self._lin_do = LinguisticInterventionGraph(cfgs)
        # self.vocab_do = LinguisticInterventionVocab(cfgs["dataset"]["causal_feature_path"])

        self.vis_norm = nn.LayerNorm(768)
        self.lin_norm = nn.LayerNorm(768)
        # self.norm_vocab = nn.LayerNorm(768)
    
    def lin_do(self, m):
        m = self._lin_do(m) + m
        m = self.lin_norm(m)
        return m
    
    def vis_do(self, x, m):
        vis_m = self._vis_do(x, m)
        m = vis_m + m
        m = self.vis_norm(m)
        return m

    def vocab_lin_do(self, x):
        x = self.norm_vocab(self.vocab_do(x) + x)
        return x


if __name__ == "__main__":
    do = LinguisticInterventionGraph()
