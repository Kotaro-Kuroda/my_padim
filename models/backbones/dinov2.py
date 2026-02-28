import torch
import torch.nn as nn


class DINOv2Backbone(nn.Module):
    def __init__(self, backbone):
        super(DINOv2Backbone, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', backbone)
        self.total_output_dim = self.model.embed_dim
        self.num_register_tokens = getattr(self.model, 'num_register_tokens', 0)

    def forward(self, x):
        with torch.no_grad():
            x = self.model.prepare_tokens_with_masks(x)

            for blk in self.model.blocks:
                x = blk(x)
            x = self.model.norm(x)
        x = x[:, 1 + self.num_register_tokens:, :]
        B, N, C = x.size()
        H = W = int(N**0.5)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return [x]
