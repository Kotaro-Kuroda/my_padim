

def backbone(backbone_name):
    if backbone_name.startswith("resnet"):
        from .resnet import ResNetBackbone
        return ResNetBackbone(backbone_name)
    elif backbone_name.startswith("dinov2"):
        from .dinov2 import DINOv2Backbone
        return DINOv2Backbone(backbone_name)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
