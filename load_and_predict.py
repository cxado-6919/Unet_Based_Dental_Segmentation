import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from modules.UNet import UNet

# Test 시 사용할 albumentations 기반 transform (data.py의 transform과 유사하게 구성)
test_transform = Compose([
    Resize(256, 256),
    Normalize(mean=[0.3439, 0.3439, 0.3439], std=[0.2127, 0.2127, 0.2127]),
    ToTensorV2()
])

# 테스트 이미지 전용 데이터셋 클래스 (마스크 없이 이미지만 로드)
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_ids = os.listdir(images_dir)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, image_id)
        # 이미지를 RGB로 읽어온 후 numpy 배열로 변환
        image = np.array(Image.open(image_path).convert("RGB"))
        if self.transform:
            # albumentations transform은 dict 형태로 결과를 반환
            image = self.transform(image=image)['image']
        return image

# 모델 생성 및 체크포인트 불러오기
num_classes = 32
model = UNet(in_channels=3, out_channels=num_classes)
checkpoint_path = "Unet_segmentation_model.pth"  # main.py에서 저장한 checkpoint 파일 이름
state_dict = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 테스트 이미지 디렉토리 (사용자 경로로 수정 필요)
images_dir = r"Your_path_here"  
test_dataset = TestDataset(images_dir=images_dir, transform=test_transform)
print("Test image IDs:", test_dataset.image_ids)

# 무작위로 한 샘플 선택하여 예측 수행
rand_idx = random.randint(0, len(test_dataset) - 1)
print(f"Random test sample index: {rand_idx}")
image = test_dataset[rand_idx]
image_tensor = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)

# 모델 출력에서 가장 높은 확률을 갖는 클래스 선택
pred_mask = torch.argmax(output, dim=1).squeeze(0)

# 정규화를 원래 픽셀 값으로 복원하는 함수 (visualize.py의 unnormalize_image와 유사)
def unnormalize(img_tensor, mean, std):
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array(mean)
    std = np.array(std)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    return img_np

mean = [0.3439, 0.3439, 0.3439]
std = [0.2127, 0.2127, 0.2127]
img_np = unnormalize(image, mean, std)
pred_mask_np = pred_mask.cpu().numpy()

# 원본 이미지, 예측 마스크, 그리고 오버레이 이미지 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img_np)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(pred_mask_np, cmap='jet', vmin=0, vmax=num_classes-1)
plt.title("Predicted Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_np)
plt.imshow(pred_mask_np, cmap='jet', alpha=0.5, vmin=0, vmax=num_classes-1)
plt.title("Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()