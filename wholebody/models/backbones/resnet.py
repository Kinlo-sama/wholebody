import torch.nn as nn
from torchvision.models.resnet import resnet50, resnet101
from wholebody.core.registry import BACKBONES

@BACKBONES.register("ResNet")
class ResNetBackbone(nn.Module):
    def __init__(self, depth: int = 50, pretrained: bool = False):
        super().__init__()
        
        if depth == 101:
            resnet = resnet101(pretrained=pretrained)
        elif depth == 50:
            resnet = resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")
            
        # Extraemos las capas de torchvision con los mismos nombres que espera MMPose
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)
