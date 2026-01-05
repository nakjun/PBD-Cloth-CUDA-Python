import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import os
import glob
from tqdm import tqdm

# 1. 데이터셋 클래스 (기존과 동일)
class ClothCollisionDataset(Dataset):
    def __init__(self, data_dir):
        # 해당 디렉토리 내의 모든 npz 파일 검색
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        print(f"📂 Found {len(self.file_list)} data files in '{data_dir}'.")
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        try:
            data = np.load(self.file_list[idx])
            
            # [Input 1] 속도 (N, 3)
            vel = data['vel'] 
            
            # [Input 2] 기하 정보 (N, 1)
            # 데이터 저장 방식에 따라 shape이 다를 수 있으므로 안전하게 reshape
            geo = data['geo'].reshape(-1, 1) 
            
            # [Feature Fusion] (N, 4) -> [vx, vy, vz, strain]
            features = np.hstack((vel, geo))
            
            # [Label] 침투 깊이 (Binary Classification)
            penetration = data['label'] 
            label = (penetration > 0.001).astype(np.float32) 
            
            return torch.FloatTensor(features), torch.FloatTensor(label)
        except Exception as e:
            print(f"❌ Error loading {self.file_list[idx]}: {e}")
            # 에러 발생 시 0으로 채운 더미 데이터 반환 (학습 중단 방지)
            return torch.zeros((1, 4)), torch.zeros((1,))

# 2. 모델 정의 (기존과 동일)
class CollisionPredictor(nn.Module):
    def __init__(self):
        super(CollisionPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid() 
        )
        
    def forward(self, x):
        return self.net(x)

def transfer_train():
    # ---------------------------------------------------------
    # [설정] 전이 학습 파라미터
    # ---------------------------------------------------------
    # 1. 학습에 사용할 데이터셋 폴더 리스트 (여기에 새 데이터 경로 추가)
    DATA_DIRS = [
        "../dataset_curtain_128",   # 기존 데이터 (Scene 1)
        "../dataset_flag_128",    # 새로운 데이터 (Scene 2)
        "../dataset_pin_128"           # 새로운 데이터 (Scene 3)
    ]
    
    PRETRAINED_MODEL_PATH = "best_model_v2.pth" # 기존 학습된 모델 경로
    SAVE_MODEL_PATH = "best_model_adapted.pth"  # 전이 학습 후 저장할 모델명
    
    BATCH_SIZE = 1 
    LR = 0.0001 # Fine-tuning을 위해 학습률을 낮춤 (0.001 -> 0.0001)
    EPOCHS = 5  # 적응(Adaptation)은 적은 에폭으로도 충분할 수 있음
    
    # ---------------------------------------------------------
    # [데이터셋 병합] 여러 폴더의 데이터를 하나로 합침
    # ---------------------------------------------------------
    datasets = []
    for d_dir in DATA_DIRS:
        if os.path.exists(d_dir):
            datasets.append(ClothCollisionDataset(d_dir))
        else:
            print(f"⚠️ Warning: Directory '{d_dir}' not found. Skipping...")
    
    if not datasets:
        print("❌ Error: No valid datasets found!")
        return

    # ConcatDataset으로 병합
    combined_dataset = ConcatDataset(datasets)
    print(f"🔥 Total Training Samples: {len(combined_dataset)}")
    
    dataloader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # ---------------------------------------------------------
    # [모델 로드 & 초기화]
    # ---------------------------------------------------------
    model = CollisionPredictor().cuda()
    
    # if os.path.exists(PRETRAINED_MODEL_PATH):
    #     print(f"📥 Loading pretrained weights from '{PRETRAINED_MODEL_PATH}'...")
    #     # 기존 모델 로드 (Key Mismatch 방지 로직 포함)
    #     checkpoint = torch.load(PRETRAINED_MODEL_PATH)
        
    #     # 만약 state_dict 키에 'net.' 접두사가 있다면 제거 (이전 저장 방식 호환)
    #     new_state_dict = {k.replace("net.", ""): v for k, v in checkpoint.items()}
        
    #     # 모델에 가중치 로드 (strict=False로 유연하게 로드)
    #     try:
    #         model.net.load_state_dict(new_state_dict)
    #     except:
    #         # 구조가 다를 경우 전체 로드 시도
    #         model.load_state_dict(checkpoint)
            
    #     print("✅ Pretrained weights loaded successfully.")
    # else:
    #     print(f"⚠️ Warning: Pretrained model '{PRETRAINED_MODEL_PATH}' not found. Starting from scratch.")

    # ---------------------------------------------------------
    # [학습 루프]
    # ---------------------------------------------------------
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("🚀 Transfer Learning Start...")
    model.train()
    
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        total_loss = 0
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
        
        for i, (features, label) in progress_bar:
            # 데이터 유효성 검사 (Shape이 이상하면 스킵)
            if features.shape[0] == 0: continue

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
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"📉 Best adapted model updated! Saved to '{SAVE_MODEL_PATH}' (loss={best_loss:.6f})")

if __name__ == "__main__":
    transfer_train()