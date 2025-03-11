# Dropout층을 추가한 U-net 클래스 모듈
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, features=[64, 128, 256, 512]):
        super().__init__()
        self.downsampling = nn.ModuleList()
        self.upsampling = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        prev_channels = in_channels
        for feature in features:
            self.downsampling.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            prev_channels = feature

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1] * 2, features[-1] * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        for feature in reversed(features):
            self.upsampling.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.upsampling.append(
                nn.Sequential(
                    nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downsampling:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.upsampling), 2):
            x = self.upsampling[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                _, _, H, W = x.shape
                skip = transforms.CenterCrop([H, W])(skip)
            x = torch.cat((skip, x), dim=1)
            x = self.upsampling[idx + 1](x)

        return self.final(x)
