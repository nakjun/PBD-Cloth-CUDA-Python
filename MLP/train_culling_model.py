import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import os
import glob
from tqdm import tqdm

# ==========================================
# [Configuration] 데이터 통계값 (Hardcoded)
# ==========================================
# 방금 뽑은 통계치를 여기에 입력합니다.
STATS = {
    'vel_mean': -0.097470,
    'vel_std': 0.458327,
    'geo_mean': 1.141827,
    'geo_std': 0.201266
}

# ==========================================
# 1. 데이터셋 클래스 (Standardization 적용)
# ==========================================
class ClothCollisionDataset(Dataset):
    def __init__(self, data_dir):
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        print(f"📂 Loading '{data_dir}': Found {len(self.file_list)} files.")
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        try:
            data = np.load(self.file_list[idx])
            
            # Raw Data Load
            vel = data['vel'] 
            geo = data['geo'].reshape(-1, 1) 
            
            # [핵심 개선] Z-Score Normalization (표준화)
            # 공식: (Value - Mean) / Std
            vel = (vel - STATS['vel_mean']) / (STATS['vel_std'] + 1e-6)
            geo = (geo - STATS['geo_mean']) / (STATS['geo_std'] + 1e-6)
            
            # Feature Fusion -> (N, 4)
            features = np.hstack((vel, geo))
            
            # Label
            penetration = data['label'] 
            # Safety Margin: 두께 0.1의 1% (0.001) 보다 가까우면 위험
            label = (penetration > 0.001).astype(np.float32) 
            
            return torch.FloatTensor(features), torch.FloatTensor(label)
        except Exception as e:
            print(f"❌ Error loading {self.file_list[idx]}: {e}")
            return torch.zeros((1, 4)), torch.zeros((1,))

# ==========================================
# 2. Pruning 모델 (배치 정규화 포함)
# ==========================================
class CollisionPruningModel(nn.Module):
    def __init__(self):
        super(CollisionPruningModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1),
            nn.Sigmoid() 
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 비대칭 손실 함수 (Asymmetric Loss)
# ==========================================
class PruningLoss(nn.Module):
    def __init__(self, miss_penalty=10.0):
        super(PruningLoss, self).__init__()
        self.miss_penalty = miss_penalty
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        loss = self.bce(pred, target)
        # 충돌(1)을 놓치면(pred<target) 페널티 부여
        weights = target * self.miss_penalty + (1 - target)
        return (loss * weights).mean()

# ==========================================
# 4. 학습 루프
# ==========================================
def train():
    # --- 설정 ---
    DATA_DIRS = [
        "../dataset_curtain_128", 
        "../dataset_flag_128", 
        "../dataset_pin_128"
    ]
    
    BATCH_SIZE = 4
    LR = 0.001
    EPOCHS = 15
    # [수정] 모델이 너무 겁먹지 않도록 페널티를 50 -> 10으로 완화
    MISS_PENALTY = 10.0 

    # --- 데이터셋 병합 ---
    datasets = []
    for d_dir in DATA_DIRS:
        if os.path.exists(d_dir):
            datasets.append(ClothCollisionDataset(d_dir))
    
    if not datasets:
        print("❌ No datasets found!")
        return

    full_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- 모델 준비 ---
    model = CollisionPruningModel().cuda()
    criterion = PruningLoss(miss_penalty=MISS_PENALTY) 
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    print(f"🚀 Training Start (Penalty: x{MISS_PENALTY})...")
    print(f"📊 Using Stats: Vel(Mean={STATS['vel_mean']:.4f}, Std={STATS['vel_std']:.4f})")
    
    model.train()
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        total_loss = 0
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
        
        for i, (features, label) in progress_bar:
            x = features.view(-1, 4).cuda()
            y = label.view(-1, 1).cuda()

            if x.shape[0] == 0: continue

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if i % 10 == 0:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_epoch_loss = total_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"==== Epoch {epoch+1} Avg Loss: {avg_epoch_loss:.6f} | LR: {current_lr:.6f} ====")
        
        scheduler.step(avg_epoch_loss)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            # 정규화된 모델 저장 (파일명 변경)
            torch.save(model.state_dict(), "best_model_norm.pth") 
            print(f"📉 Saved 'best_model_norm.pth' (Loss: {best_loss:.6f})")

if __name__ == "__main__":
    train()