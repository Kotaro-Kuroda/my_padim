
import random
from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import backbone


class PaDiM(nn.Module):
    def __init__(self, backbone_name, embed_dim=100):
        super(PaDiM, self).__init__()
        self.backbone = backbone.backbone(backbone_name)
        self.total_output_dim = self.backbone.total_output_dim
        self.embed_dim = embed_dim

        random.seed(1024)
        torch.manual_seed(1024)

        self.idx = torch.tensor(sample(range(0, self.total_output_dim), self.embed_dim))

    def embedding_concat(self, features):
        head = features[0]
        B, C, H, W = head.size()
        for feature in features[1:]:
            feature = F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=False)
            head = torch.cat((head, feature), dim=1)
        return head

    def forward_feature(self, x):
        self.backbone.eval()
        with torch.no_grad():
            outputs = self.backbone(x)
        x = self.embedding_concat(outputs)
        x = torch.index_select(x, dim=1, index=self.idx)
        return x

    def mahalanobis_distance(self, x, mean, cov_inv):
        B, C, H, W = x.size()
        x = x.view(B, C, H * W)
        delta = x - mean
        md = torch.einsum('cdx, bdx, bcx -> bx', cov_inv, delta, delta)
        md = torch.sqrt(md)
        md = md.view(B, H, W)
        return md

    def forward(self, x):
        x = self.forward_feature(x)
        return x
