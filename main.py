import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

# 처리 디바이스 지정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# U-Net 클래스
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downsampling = nn.ModuleList()
        self.upsampling = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downsampling.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU()
                )
            )
            in_channels=feature

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features[-1]*2, features[-1]*2, kernel_size=3, padding=1),
            nn.ReLU()
        )

        for feature in reversed(features):
            self.upsampling.append(
                    nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.upsampling.append(
                nn.Sequential(
                    nn.Conv2d(feature*2, feature, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU()
                )
            )
        
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connection = []

        for down_layer in self.downsampling:
            x = down_layer(x)
            skip_connection.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)

        skip_connection = skip_connection[::-1]

        for idx in range(0, len(self.upsampling), 2):
            x = self.upsampling[idx](x)
            skip_connection = skip_connection[idx // 2]

            if x.shape != skip_connection.shape:
                skip_connection = self.crop(skip_connection, x)

            x = torch.cat((skip_connection, x), dim=1)
            x = self.upsampling[idx+1](x)
        
        output = self.final(x)

        return output

    def crop(self, enc_features, x):

        _, _, H, W = x.shape
        transform = transforms.CenterCrop([H,W])
        return transform(enc_features)
    

# 데이터셋 불러오기(X-ray 이미지, 마스크)

class XrayDataset(torch.utils.Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_names = sorted(os.listdir(image_dir))
        self.mask_names = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # transform이 있다면 적용(일단 적용하지 않고 학습시켜볼 예정)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask).long()

        return image, mask    
    

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

if __name__ == '__main__':
    image_dir = "엑스레이 이미지 경로"
    mask_dir = "마스크 이미지 경로"

    dataset = XrayDataset(image_dir, mask_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    num_classes = 33
    model = UNet()

    model = UNet
    learning_rate = 0.001
    loss_function = nn.CrossEntropyLoss()


        
