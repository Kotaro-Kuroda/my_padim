
import torch
import torch.nn as nn
from torchvision import models


class ResNetBackbone(nn.Module):
    def __init__(self, backbone):
        super(ResNetBackbone, self).__init__()

        model = getattr(models, backbone)(pretrained=True)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.total_output_dim = sum([self._layer_out_channels(layer) for layer in [self.layer1, self.layer2, self.layer3]])

    def _layer_out_channels(self, layer):
        block = layer[-1]
        if hasattr(block, "bn3"):   # Bottleneck (resnet50/101...)
            return block.bn3.num_features
        if hasattr(block, "bn2"):   # BasicBlock (resnet18/34)
            return block.bn2.num_features
        raise ValueError("Unknown block type")

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        x = self.layer1(x)
        outputs.append(x)
        x = self.layer2(x)
        outputs.append(x)
        x = self.layer3(x)
        outputs.append(x)
        return outputs
