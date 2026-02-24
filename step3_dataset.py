import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm



class CervicalDataset(Dataset):
    def __init__(self, patient_list_file, data_dir, transform=None, cache_to_ram=True):
        with open(patient_list_file, 'r') as f:
            self.patient_ids = [line.strip() for line in f.readlines()]
        
        self.data_dir = data_dir
        self.transform = transform
        self.cache_to_ram = cache_to_ram
        self.num_slices = 128
        
        self.image_cache = {}
        self.mask_cache = {}

        if self.cache_to_ram:
            print(f"🧠 {patient_list_file} 데이터를 RAM에 로딩 중...")
            for pid in tqdm(self.patient_ids):
                img = np.load(os.path.join(data_dir, f"{pid}_img.npy")).astype(np.float32)
                mask = np.load(os.path.join(data_dir, f"{pid}_mask.npy")).astype(np.uint8)
                # 라벨 8(T1), 9(T2)는 배경(0)으로 밀기
                mask = np.where(mask > 7, 0, mask)
                
                self.image_cache[pid] = img
                self.mask_cache[pid] = mask
            print(f"✅ 로딩 완료!")

    def __len__(self):
        return len(self.patient_ids) * self.num_slices

    def __getitem__(self, idx):
        patient_idx = idx // self.num_slices
        slice_idx = idx % self.num_slices
        pid = self.patient_ids[patient_idx]

        # 1. 데이터 가져오기
        if self.cache_to_ram:
            image = self.image_cache[pid][slice_idx]
            mask = self.mask_cache[pid][slice_idx]
        else:
            image = np.load(os.path.join(self.data_dir, f"{pid}_img.npy"))[slice_idx]
            mask = np.load(os.path.join(self.data_dir, f"{pid}_mask.npy"))[slice_idx]
            # [Fix] 캐싱 안 할 때도 라벨 정리가 필요함
            mask = np.where(mask > 7, 0, mask).astype(np.uint8)

        # 2. Augmentation (image: H, W / mask: H, W)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image'] # (1, 224, 224) 텐서
            mask = augmented['mask']   # (224, 224) 텐서
        else:
            # transform이 없을 경우를 대비한 기본 텐서화
            image = torch.from_numpy(image).unsqueeze(0) # (1, 224, 224)
            mask = torch.from_numpy(mask).long()

        # 3. Z-position 채널 추가 (0 ~ 1)
        z_pos = slice_idx / (self.num_slices - 1)
        # image.shape[1:]를 써서 H, W 크기만 가져와 2D 평면 생성
        z_channel = np.full(image.shape[1:], z_pos, dtype=np.float32)
        z_tensor = torch.from_numpy(z_channel).unsqueeze(0) # (1, 224, 224)
        
        # 4. 채널 결합 (Image + Z) -> (2, 224, 224)
        image = torch.cat([image, z_tensor], dim=0)

        # 마스크 타입 보정
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        mask = mask.long()

        return image, mask

# --- 데이터 증강(Augmentation) 정의 예시 ---
def get_transforms(mode='train'):
    if mode == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),      # 50% 확률로 좌우 반전
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5), # 미세한 회전/이동
            A.RandomBrightnessContrast(p=0.2), # 밝기 조절
            # Normalize는 모델의 pre-trained 가중치에 따라 결정 (기본은 ToTensorV2)
            ToTensorV2()
        ])
    else:
        # 검증(Val) 시에는 증강 없이 텐서 변환만
        return A.Compose([
            ToTensorV2()
        ])