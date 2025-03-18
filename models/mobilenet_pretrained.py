import torch
import torch.nn as nn
from torchvision import models

class MobilenetModel(nn.Module):
    def __init__(self, num_classes=5):
        super(MobilenetModel, self).__init__()
        # Load a pre-trained MobileNetV2 model (small model)
        self.model = models.mobilenet_v2( pretrained=True)

        # Freeze pre-trained layers
        for param in self.model.parameters():
           param.requires_grad = False

        # Modify the final classification layer
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, 512)
        self.out_linear = nn.Linear(512, num_classes)

    def forward(self, x):
        c = self.model(x)
        c = torch.nn.functional.relu6(c)
        # return torch.nn.functional.relu(self.out_linear(c))
        return torch.exp(self.out_linear(c))