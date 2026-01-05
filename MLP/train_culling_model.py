import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from tqdm import tqdm

# 1. 데이터셋 클래스 정의 (Geometry Feature 추가)
class ClothCollisionDataset(Dataset):
    def __init__(self, data_dir):
        # v2 데이터 폴더로 경로가 맞는지 꼭 확인하게!
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        print(f"📂 Found {len(self.file_list)} data files in '{data_dir}'.")
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # npz 로드
        data = np.load(self.file_list[idx])
        
        # [Input 1] 속도 (N, 3)
        vel = data['vel'] 
        
        # [Input 2] 기하 정보 (N, 1) - 우리가 새로 추가한 핵심 Feature!
        # 만약 geo가 (N,) 형태로 저장되었다면 reshape(-1, 1)이 필요할 수 있음
        geo = data['geo'].reshape(-1, 1) 
        
        # [Feature Fusion] 속도와 기하 정보를 합침 -> (N, 4)
        # 이제 입력 벡터는 [vx, vy, vz, strain] 형태가 됨
        features = np.hstack((vel, geo))
        
        # [Label] 침투 깊이 (Binary Classification)
        penetration = data['label'] 
        label = (penetration > 0.001).astype(np.float32) 
        
        return torch.FloatTensor(features), torch.FloatTensor(label)

# 2. 모델 정의 (입력 차원 변경: 3 -> 4)
class CollisionPredictor(nn.Module):
    def __init__(self):
        super(CollisionPredictor, self).__init__()
        # 입력: 4 (vx, vy, vz, compression_ratio)
        # 출력: 1 (충돌 확률)
        self.net = nn.Sequential(
            nn.Linear(4, 32), # <--- 여기가 핵심 변경점일세!
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() 
        )
        
    def forward(self, x):
        return self.net(x)

def train():
    # 설정
    # 아까 cloth.py에서 저장한 폴더명으로 수정했네 (v2)
    DATA_DIR = "../dataset_curtain_128" 
    
    BATCH_SIZE = 1 
    LR = 0.001
    EPOCHS = 10
    
    # 데이터셋 경로 존재 확인
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Data directory '{DATA_DIR}' not found!")
        return

    # 데이터 로더
    dataset = ClothCollisionDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 모델 & 최적화
    model = CollisionPredictor().cuda()
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("🚀 Training Start with Geometry Features...")
    model.train()
    
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        total_loss = 0
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
        
        for i, (features, label) in progress_bar:
            # features shape: [1, N, 4] -> [N, 4]로 펼침
            # label shape: [1, N] -> [N, 1]로 펼침

            x = features.view(-1, 4).cuda()
            y = label.view(-1, 1).cuda()

            # Forward
            pred = model(x)
            loss = criterion(pred, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            avg_loss = total_loss / (i + 1)
            progress_bar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

        avg_epoch_loss = total_loss / len(dataloader)
        print(f"==== Epoch {epoch} Average Loss: {avg_epoch_loss:.6f} ====")

        # Best Model 저장
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), "best_model_v2.pth") # 파일명도 v2로 바꿈
            print(f"📉 Best model updated! Saved to 'best_model_v2.pth' (loss={best_loss:.6f})")

if __name__ == "__main__":
    train()