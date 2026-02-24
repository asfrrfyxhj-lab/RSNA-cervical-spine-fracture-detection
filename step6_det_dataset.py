import os
import numpy as np
import pandas as pd
import torch
import ast
from torch.utils.data import Dataset

class SpineYoloDataset(Dataset):
    def __init__(self, df, base_dir, transform=None):
        self.df = df
        self.base_dir = base_dir
        self.transform = transform
        self.img_size = 512 # [수정] 512px 해상도

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.base_dir, row['file_path'])
        
        # 1. 4채널 로드
        try:
            with np.load(file_path) as loaded:
                img = loaded['data'].astype(np.float32) # (4, 512, 512)
                h, w = img.shape[1], img.shape[2]
        except:
            # 로드 실패 시 더미 데이터
            img = np.zeros((4, self.img_size, self.img_size), dtype=np.float32)
            h, w = self.img_size, self.img_size
        
        # Albumentations용 채널 변환 (C, H, W) -> (H, W, C)
        img_hwc = img.transpose(1, 2, 0)

        # 2. 라벨 파싱 (강화된 로직)
        boxes = []
        
        # 골절 여부 확인 (문자/숫자 호환)
        try:
            is_fracture = int(float(row['fracture'])) == 1
        except:
            is_fracture = False

        if is_fracture and pd.notna(row['bbox']):
            try:
                raw_boxes = row['bbox']
                # [중요] Numpy 포맷 문자열 처리
                if isinstance(raw_boxes, str):
                    if 'np.float64' in raw_boxes:
                        raw_boxes = raw_boxes.replace('np.float64(', '').replace(')', '')
                    raw_boxes = ast.literal_eval(raw_boxes)
                
                if isinstance(raw_boxes, list):
                    for box in raw_boxes:
                        if len(box) == 4:
                            # 512 기준 정규화된 좌표가 들어있다고 가정
                            x_n, y_n, w_n, h_n = box
                            
                            # YOLO 포맷 변환 (Center X, Center Y, W, H)
                            cx = x_n + w_n / 2
                            cy = y_n + h_n / 2
                            
                            # 뼈 클래스 (0~6)
                            try:
                                cls = int(float(row['ver_level'])) - 1
                            except:
                                continue

                            if 0 <= cls < 7:
                                boxes.append([cls, cx, cy, w_n, h_n])
            except Exception:
                pass # 파싱 에러 시 박스 없음 처리

        # 3. Augmentation (옵션)
        if self.transform:
            # boxes 포맷: [cls, cx, cy, w, h] -> Albumentations는 [cx, cy, w, h, cls] 등을 요구하므로 변환 주의
            # 여기서는 복잡함을 피하기 위해 Transform은 생략하거나, 이미지에만 적용한다고 가정
            # (4채널 Augmentation은 설정이 까다로우므로 일단 이미지 텐서 변환만 수행)
            pass
        
        # 텐서 변환
        img_tensor = torch.from_numpy(img).float()
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))

        # [핵심] Trainer 호환 반환값: (이미지, 박스, 원본크기, 파일경로)
        return img_tensor, boxes_tensor, (h, w), file_path

def yolo_collate_fn(batch):
    """
    Ultralytics DetectionTrainer 호환 Collate Function
    """
    images, box_lists, shapes, paths = zip(*batch)
    
    # 이미지 스택
    images = torch.stack(images, 0)
    
    # 박스 병합 (Batch Index 추가)
    # boxes: [batch_idx, cls, cx, cy, w, h]
    new_boxes = []
    for i, boxes in enumerate(box_lists):
        if boxes.shape[0] > 0:
            idx_col = torch.full((boxes.shape[0], 1), i, dtype=torch.float32)
            combined = torch.cat([idx_col, boxes], dim=1)
            new_boxes.append(combined)
            
    targets = torch.cat(new_boxes, 0) if new_boxes else torch.zeros((0, 6))
    
    # Trainer가 요구하는 Flat Dictionary 구조
    return {
        'img': images,
        'batch_idx': targets[:, 0],
        'cls': targets[:, 1].view(-1, 1),
        'bboxes': targets[:, 2:],
        'device': None,
        # 검증기(Validator) 필수 메타데이터
        'ori_shape': list(shapes),
        'resized_shape': list(shapes),
        'im_file': list(paths),
        'ratio_pad': [((1.0, 1.0), (0.0, 0.0)) for _ in range(len(images))]
    }