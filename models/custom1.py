import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class C3K2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class CustomModel1(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial convolution with reduced channels
        self.conv1 = Conv(3, 24, k=3, s=2)
        
        # C3K2 blocks with reduced channels and fewer bottlenecks
        self.c3k2_1 = C3K2(24, 48, n=2)  # Reduced from 3
        self.c3k2_2 = C3K2(48, 96, n=3)   # Reduced from 6
        self.c3k2_3 = C3K2(96, 192, n=4)  # Reduced from 9
        self.c3k2_4 = C3K2(192, 384, n=2) # Reduced from 3
        
        # Downsampling layers with adjusted channels
        self.down1 = Conv(48, 48, k=3, s=2)
        self.down2 = Conv(96, 96, k=3, s=2)
        self.down3 = Conv(192, 192, k=3, s=2)
        self.down4 = Conv(384, 384, k=3, s=2)
        
        # Classification head with reduced dimensions
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(384, 512)  # Reduced from 1024
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 5)
        
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        
        # C3K2 blocks with downsampling
        x = self.c3k2_1(x)
        x = self.down1(x)
        
        x = self.c3k2_2(x)
        x = self.down2(x)
        
        x = self.c3k2_3(x)
        x = self.down3(x)
        
        x = self.c3k2_4(x)
        x = self.down4(x)
        
        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Apply exponential to output
        x = torch.exp(x)
        
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)