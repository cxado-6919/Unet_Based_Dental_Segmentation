import os
import torch
from torch.utils.data import DataLoader, random_split
from data import XRayDataset, get_train_transform, get_val_test_transform
from train import train_model
from utils import set_seed

# seed 설정 (재현성을 위해)
set_seed(42)

# 데이터셋 경로 설정 (자신의 경로에 맞게 수정)
images_dir = '/home/elicer/.cache/kagglehub/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images/versions/1/Teeth Segmentation PNG/d2/img'
masks_dir = '/home/elicer/.cache/kagglehub/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images/versions/1/Teeth Segmentation PNG/d2/masks_machine'

# 데이터셋 생성 및 분할
full_dataset = XRayDataset(images_dir, masks_dir, transform=get_train_transform())
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# 검증/테스트 데이터셋에는 augmentation 없이 transform 적용
val_dataset.dataset.transform = get_val_test_transform()
test_dataset.dataset.transform = get_val_test_transform()

# DataLoader 구성
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 모델 생성 (이미 modules/UNet.py에 정의된 UNet 사용)
from modules.UNet import UNet
num_classes = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=num_classes).to(device)

# 학습 실행
num_epochs = 200
checkpoint_path = "best_model_checkpoint.pth"
model = train_model(model, train_loader, val_loader, num_epochs, device, checkpoint_path=checkpoint_path)

# 테스트 평가
model.eval()
test_loss = 0.0
from losses import CombinedLoss
criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0)
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        test_loss += loss.item()
avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")
