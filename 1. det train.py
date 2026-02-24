import os
# --- [1. OpenMP 에러 방지] ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast
import yaml

# Ultralytics 공식 트레이너 관련
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

# ==========================================
# 2. 설정 (Configuration)
# ==========================================
CONFIG = {
    'base_dir': r'C:\conda\3. Project\rsna-2022-cervical-spine-fracture-detection\golden_dataset',
    # [수정] 512용 매니페스트 경로로 변경
    'manifest_path': r'C:\conda\3. Project\rsna-2022-cervical-spine-fracture-detection\golden_dataset\det_train_manifest_512_folds.csv',
    
    # [수정] 요청하신 yolo26s (없으면 v8s로 대체됨)
    'pretrained_model': 'yolo26n.pt', 
    'save_dir_root': './runs/rsna_5fold_512', # 512 결과 저장소
    'img_size': 512,    # [수정] 512 해상도
    'batch_size': 16,   # [주의] 512로 커졌으므로 배치 사이즈를 32->16으로 줄임 (OOM 방지)
    'epochs': 50,       # 충분한 학습
    'patience': 10,
    'workers': 0,
    'folds_to_run': [1, 2, 3, 4, 5]
}

def create_dummy_structure(base_dir):
    fake_dirs = [
        os.path.join(base_dir, 'images', 'train'), os.path.join(base_dir, 'images', 'val'),
        os.path.join(base_dir, 'labels', 'train'), os.path.join(base_dir, 'labels', 'val')
    ]
    for d in fake_dirs: os.makedirs(d, exist_ok=True)

# ==========================================
# 3. 데이터셋
# ==========================================
class RSNA4ChannelDataset(Dataset):
    def __init__(self, df, base_dir):
        self.df = df
        self.base_dir = base_dir
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 512 데이터 경로인지 확인 (manifest가 이미 정확한 경로를 가지고 있어야 함)
        full_path = os.path.join(self.base_dir, row['file_path'])
        
        # 1. 이미지 로드
        try:
            with np.load(full_path) as loaded:
                img = loaded['data'].astype(np.float32)
                h, w = img.shape[1], img.shape[2] 
        except:
            img = np.zeros((4, CONFIG['img_size'], CONFIG['img_size']), dtype=np.float32)
            h, w = CONFIG['img_size'], CONFIG['img_size']

        # 2. 박스 로드 (Numpy 포맷 문자열 처리 포함)
        boxes = [] 
        
        try:
            fracture_flag = int(float(row['fracture']))
        except:
            fracture_flag = 0
            
        if fracture_flag == 1 and pd.notna(row['bbox']):
            try:
                raw_boxes = row['bbox']
                # np.float64 문자열 제거 로직
                if isinstance(raw_boxes, str):
                    if 'np.float64' in raw_boxes:
                        raw_boxes = raw_boxes.replace('np.float64(', '').replace(')', '')
                    raw_boxes = ast.literal_eval(raw_boxes)
                
                if isinstance(raw_boxes, list):
                    for box in raw_boxes:
                        # 매니페스트에 이미 0~1 정규화된 좌표가 들어있다고 가정
                        if len(box) == 4:
                            x, y, w_box, h_box = box
                            cx, cy = x + w_box/2, y + h_box/2
                            
                            try:
                                cls = int(float(row['ver_level'])) - 1 
                            except:
                                cls = -1
                                
                            if 0 <= cls < 7:
                                boxes.append([cls, cx, cy, w_box, h_box])
            except Exception as e:
                pass
        
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))
        img_tensor = torch.from_numpy(img)
        
        # Trainer 호환 반환 (이미지, 박스, 원본크기, 경로)
        return img_tensor, boxes, (h, w), full_path

def trainer_collate_fn(batch):
    images, box_lists, shapes, paths = zip(*batch)
    images = torch.stack(images, 0)
    
    new_boxes = []
    for i, boxes in enumerate(box_lists):
        if boxes.shape[0] > 0:
            idx_col = torch.full((boxes.shape[0], 1), i, dtype=torch.float32)
            combined = torch.cat([idx_col, boxes], dim=1)
            new_boxes.append(combined)
    targets = torch.cat(new_boxes, 0) if new_boxes else torch.zeros((0, 6))
    
    return {
        'img': images,
        'batch_idx': targets[:, 0],
        'cls': targets[:, 1].view(-1, 1),
        'bboxes': targets[:, 2:],
        'device': None,
        'ori_shape': list(shapes),
        'resized_shape': list(shapes),
        'im_file': list(paths),
        'ratio_pad': [((1.0, 1.0), (0.0, 0.0)) for _ in range(len(images))] 
    }

# ==========================================
# 4. 커스텀 트레이너
# ==========================================
class RSNATrainer(DetectionTrainer):
    def __init__(self, overrides=None, current_fold=1):
        self.current_fold = current_fold
        super().__init__(overrides=overrides)

    def build_dataset(self, img_path, mode="train", batch=None):
        df = pd.read_csv(CONFIG['manifest_path'])
        if mode == "train":
            sub_df = df[df['fold'] != self.current_fold].reset_index(drop=True)
        else:
            sub_df = df[df['fold'] == self.current_fold].reset_index(drop=True)
            
        print(f"Dataset Build ({mode}): {len(sub_df)} samples loaded.")
        return RSNA4ChannelDataset(sub_df, CONFIG['base_dir'])

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        dataset = self.build_dataset(dataset_path, mode)
        # 512는 메모리를 많이 먹으므로 num_workers=0 유지
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"),
                            num_workers=0, collate_fn=trainer_collate_fn, pin_memory=True)
        loader.reset = lambda: None
        return loader

    def get_model(self, cfg=None, weights=None, verbose=True):
        # 모델 파일 확인 및 Fallback 로직
        if weights and isinstance(weights, str) and os.path.exists(weights):
            target_weights = weights
        else:
            # yolo26s가 없으면 yolov8s(Small)로 대체 (Nano보다 성능 좋음)
            print("⚠️ 지정된 모델 파일이 없어 'yolov8s.pt'로 대체합니다.")
            target_weights = 'yolov8s.pt'
            
        temp_yolo = YOLO(target_weights)
        model = temp_yolo.model
        
        m = model.model[0]
        if isinstance(m.conv, nn.Conv2d) and m.conv.in_channels == 3:
            old_conv = m.conv
            new_conv = nn.Conv2d(4, old_conv.out_channels, 
                                 kernel_size=old_conv.kernel_size, stride=old_conv.stride, 
                                 padding=old_conv.padding, bias=(old_conv.bias is not None))
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)
                if old_conv.bias is not None: new_conv.bias = old_conv.bias
            m.conv = new_conv
            m.c1 = 4
            print("🔧 Conv1 4채널 확장 완료")
        return model

    def plot_training_labels(self, *args, **kwargs): pass 

# ==========================================
# 5. 커스텀 메트릭 (별도 계산)
# ==========================================
def calculate_custom_metrics(model_path, val_fold, device):
    print(f"\n📊 [Fold {val_fold}] 커스텀 메트릭 계산 시작...")
    df = pd.read_csv(CONFIG['manifest_path'])
    val_df = df[df['fold'] == val_fold].reset_index(drop=True)
    val_ds = RSNA4ChannelDataset(val_df, CONFIG['base_dir'])
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    # 모델 로드 (가중치만)
    temp_trainer = RSNATrainer(overrides={'model': 'yolov8s.pt', 'data': 'rsna_dummy.yaml'}, current_fold=val_fold)
    model = temp_trainer.get_model(weights=model_path)
    model.to(device)
    model.eval()

    preds_prob = []
    targets = []

    with torch.no_grad():
        for img, boxes, _, _ in val_loader:
            img = img.to(device)
            output = model(img)
            pred_tensor = output[0].transpose(1, 2)
            
            for i in range(len(img)):
                p = pred_tensor[i]
                cls_probs = p[:, 4:]
                conf, _ = cls_probs.max(1)
                max_conf = conf.max().item()
                preds_prob.append(max_conf)
            
            for b in boxes:
                targets.append(1.0 if b.shape[0] > 0 else 0.0)

    preds_prob = np.array(preds_prob)
    targets = np.array(targets)
    
    acc = ((preds_prob > 0.5) == targets).mean()
    epsilon = 1e-15
    preds_prob = np.clip(preds_prob, epsilon, 1 - epsilon)
    log_loss = -(targets * np.log(preds_prob) + (1 - targets) * np.log(1 - preds_prob)).mean()

    print(f"   👉 Accuracy: {acc:.4f}")
    print(f"   👉 Log Loss: {log_loss:.4f}")
    return acc, log_loss

# [데이터셋 건강검진]
def check_dataset_health(df, base_dir):
    print("\n🩺 데이터셋 건강검진 시작...")
    frac_df = df[df['fracture'] == 1].reset_index(drop=True)
    if len(frac_df) == 0:
        print("   ⚠️ 경고: 골절 데이터(fracture=1)가 CSV에 없습니다.")
        return

    ds = RSNA4ChannelDataset(frac_df, base_dir)
    found_boxes = 0
    for i in range(min(500, len(ds))):
        _, boxes, _, _ = ds[i]
        if boxes.shape[0] > 0:
            found_boxes += 1
            if found_boxes == 1:
                print(f"   ✅ 첫 번째 박스 발견! (Sample {i}): {boxes[0].tolist()}")
    
    print(f"   📊 골절 데이터 500개 중 박스 파싱 성공: {found_boxes}개")
    if found_boxes == 0:
        print("   ❌ 오류: 여전히 박스가 파싱되지 않습니다. CSV 형식을 재확인하세요.")
    else:
        print("   ✅ 데이터셋 로드 정상 확인 완료! 학습을 시작합니다.")

# ==========================================
# 6. 메인 실행
# ==========================================
if __name__ == '__main__':
    create_dummy_structure(CONFIG['base_dir'])
    
    abs_base = os.path.abspath(CONFIG['base_dir'])
    dummy_yaml_path = os.path.join(CONFIG['base_dir'], "rsna_dummy.yaml")
    
    with open(dummy_yaml_path, 'w') as f:
        yaml.dump({
            'path': abs_base, 'train': 'images/train', 'val': 'images/val',
            'nc': 7, 'names': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
        }, f)

    final_results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # [시작 전 체크]
    full_df = pd.read_csv(CONFIG['manifest_path'])
    check_dataset_health(full_df, CONFIG['base_dir'])

    for fold in CONFIG['folds_to_run']:
        print(f"\n{'='*40}")
        print(f"🚀 FOLD {fold} 학습 시작")
        print(f"{'='*40}")
        save_dir = os.path.join(CONFIG['save_dir_root'], f'fold_{fold}')
        
        args = dict(
            model=CONFIG['pretrained_model'], data=dummy_yaml_path, epochs=CONFIG['epochs'],
            batch=CONFIG['batch_size'], imgsz=CONFIG['img_size'], device=0 if device == 'cuda' else 'cpu',
            project=CONFIG['save_dir_root'], name=f'fold_{fold}', exist_ok=True, workers=0,
            val=True, plots=False, amp=True, save=True, patience=CONFIG['patience']
        )

        trainer = RSNATrainer(overrides=args, current_fold=fold)
        trainer.train()
        
        best_pt = os.path.join(save_dir, 'weights', 'best.pt')
        if not os.path.exists(best_pt): best_pt = os.path.join(save_dir, 'weights', 'last.pt')
            
        acc, log_loss = calculate_custom_metrics(best_pt, fold, device)
        
        yolo_res_path = os.path.join(save_dir, 'results.csv')
        map50 = 0.0
        if os.path.exists(yolo_res_path):
            try:
                res_df = pd.read_csv(yolo_res_path)
                res_df.columns = [c.strip() for c in res_df.columns]
                map50 = res_df['metrics/mAP50(B)'].iloc[-1]
            except: pass

        final_results.append({'fold': fold, 'accuracy': acc, 'log_loss': log_loss, 'mAP50': map50})
        pd.DataFrame(final_results).to_csv(os.path.join(CONFIG['save_dir_root'], 'rsna_final_summary.csv'), index=False)

    print(f"\n✨ 모든 Fold 학습 완료!")
    print(pd.DataFrame(final_results))