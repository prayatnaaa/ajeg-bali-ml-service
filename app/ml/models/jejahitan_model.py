import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm



class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        mid_planes = max(8, in_planes // ratio)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, mid_planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_planes, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class CosineLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma: bool = True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter("sigma", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.sigma is not None:
            self.sigma.data.fill_(10.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = F.normalize(input, p=2, dim=1)
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        out = F.linear(normalized_input, normalized_weight)
        if self.sigma is not None:
            out = self.sigma * out
        return out


class MobileViTv2WithCBAMOptimized(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(MobileViTv2WithCBAMOptimized, self).__init__()

        if timm is None:
            raise ImportError("timm is required to build MobileViTv2WithCBAMOptimized")
        
        print("Loading MobileViTv2...")

        self.backbone = timm.create_model(
            "mobilevitv2_100",
            pretrained=pretrained,
            features_only=True,
        )

        print("MobileViTv2 loaded successfully.")

        in_features = self.backbone.feature_info.channels()[-1]
        self.cbam = CBAM(in_planes=in_features, ratio=16, kernel_size=7)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = CosineLinear(in_features, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        x = features[-1]
        x = self.cbam(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
