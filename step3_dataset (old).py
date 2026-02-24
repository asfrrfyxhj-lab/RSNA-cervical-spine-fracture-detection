import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm



class CervicalDataset(Dataset):
    def __init__(self, patient_list_file, data_dir, transform=None, cache_to_ram=True):
        """
        Args:
            patient_list_file (str): train_patients.txt 또는 val_patients.txt 경로
            data_dir (str): .npy 파일들이 모여있는 final_preprocessed 폴더 경로
            transform (albumentations): 데이터 증강 함수
            cache_to_ram (bool): 데이터를 RAM에 미리 다 올려둘지 여부
        """
        with open(patient_list_file, 'r') as f:
            self.patient_ids = [line.strip() for line in f.readlines()]
        
        self.data_dir = data_dir
        self.transform = transform
        self.cache_to_ram = cache_to_ram
        self.num_slices = 128 # 전처리 단계에서 맞춘 장수
        
        # RAM 캐싱을 위한 딕셔너리
        self.image_cache = {}
        self.mask_cache = {}

        if self.cache_to_ram:
            print(f"🧠 {patient_list_file} 데이터를 RAM에 로딩 중...")
            for pid in tqdm(self.patient_ids):
                img = np.load(os.path.join(data_dir, f"{pid}_img.npy")).astype(np.float32)
                mask = np.load(os.path.join(data_dir, f"{pid}_mask.npy")).astype(np.uint8)
                # ⭐ [여기에 추가!] 8번(T1) 이상의 흉추 라벨은 배경(0)으로 밀어버립니다.
                # 전처리에서 T2까지 가져오더라도, 학습 정답지는 C1~C7(1~7)만 남게 됩니다.
                mask = np.where(mask > 7, 0, mask)
                
                self.image_cache[pid] = img
                self.mask_cache[pid] = mask
            print(f"✅ 로딩 완료! (환자 수: {len(self.patient_ids)})")

    def __len__(self):
        # 전체 데이터 개수는 '환자 수 * 128장'입니다.
        return len(self.patient_ids) * self.num_slices

    def __getitem__(self, idx):
        # 1. 전체 인덱스를 '환자 번호'와 '슬라이스 번호'로 변환
        patient_idx = idx // self.num_slices
        slice_idx = idx % self.num_slices
        pid = self.patient_ids[patient_idx]

        # 2. 데이터 가져오기 (캐시 또는 디스크에서)
        if self.cache_to_ram:
            image = self.image_cache[pid][slice_idx] # (224, 224)
            mask = self.mask_cache[pid][slice_idx]   # (224, 224)
        else:
            # 캐싱 안 할 경우 여기서 파일을 직접 읽음 (느림)
            vol = np.load(os.path.join(self.data_dir, f"{pid}_img.npy"))
            image = vol[slice_idx]
            m_vol = np.load(os.path.join(self.data_dir, f"{pid}_mask.npy"))
            mask = m_vol[slice_idx]

        # 3. 데이터 증강 (Augmentation) 적용
        # albumentations은 이미지와 마스크를 동시에 변환해줍니다.
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 4. 실시간 Z-position 채널 추가 (0 ~ 1)
        z_pos = slice_idx / (self.num_slices - 1)
        z_channel = np.full(image.shape, z_pos, dtype=np.float32)
        
        # 5. 채널 결합 (Image: Ch 0, Z-pos: Ch 1) -> (2, 224, 224)
        if torch.is_tensor(image):
            # ToTensorV2가 이미 (1, 224, 224)를 만들었으므로
            # 새로운 차원을 만드는 stack 대신, 기존 채널 차원에 합치는 cat을 사용합니다.
            z_tensor = torch.tensor(z_channel) # z_channel도 (1, 224, 224) 형태가 됨
            image = torch.cat([image, z_tensor], dim=0) # (1+1, 224, 224) -> (2, 224, 224)
        else:
            # 이 부분은 transform이 없을 때(numpy 상태) 실행됨
            image = np.stack([image, z_channel], axis=0)
            image = torch.from_numpy(image)

        # 마스크를 Long 텐서로 변환 (Loss 계산용)
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask).long()

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