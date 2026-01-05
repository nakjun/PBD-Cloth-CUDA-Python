import torch
import numpy as np
import os
import sys

sys.path.append('../')
from Cloth.cloth import ClothSimulator
from tqdm import tqdm
from train_culling_model import CollisionPredictor

# 자네가 아까 정의했던 모델 클래스를 가져와야 하네. (같은 파일에 있다면 생략 가능)
# from model_structure import CollisionPredictor 

class NeuralCollisionDetector:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # [중요] 학습 코드와 동일한 구조여야 함 (Input: 4)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 32), # 속도(3) + 기하(1)
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        ).to(self.device)
        
        # 가중치 로드
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # -----------------------------------------------------------
            # [수술 집도] Key Mismatch 해결 로직
            # 학습된 모델은 'net.0.weight' 처럼 'net.'이 붙어있음.
            # 현재 self.model은 '0.weight'를 원함.
            # 따라서 딕셔너리의 키(Key)에서 'net.'을 제거해야 함.
            # -----------------------------------------------------------
            new_state_dict = {}
            for k, v in checkpoint.items():
                name = k.replace("net.", "") # 'net.' 접두사 제거
                new_state_dict[name] = v
                
            # 수정된 state_dict로 로드
            self.model.load_state_dict(new_state_dict)
            self.model.eval()
            print(f"🧠 AI Model V2 Loaded (Keys Fixed): {model_path}")
        else:
            raise FileNotFoundError(f"모델 파일이 없네: {model_path}")

    def predict(self, velocities, positions, simulator):
        """
        Input: 
            - velocities: (N, 3)
            - positions: (N, 3) 
            - simulator: 기하 정보 추출을 위한 시뮬레이터 인스턴스
        Output: (N, ) 충돌 확률
        """
        # 1. Feature Extraction (On-the-fly)
        # 시뮬레이터 함수를 재사용해 즉석에서 Strain 계산
        geo_feature = simulator.get_compression_feature(positions) # (N, 1) Numpy
        
        # 2. Feature Fusion (Vel + Geo)
        # 학습 때와 똑같이 합쳐야 하네 (N, 4)
        features = np.hstack((velocities, geo_feature))
        
        # 3. Inference
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            pred = self.model(x) # (N, 1)
            return pred.squeeze().cpu().numpy()

def main_inference_test():
    print("🧪 Starting AI Inference Test...")

    # 1. 시뮬레이션 환경 설정
    width, height = 128, 128
    sim = ClothSimulator(width, height, spacing=0.1)
    
    # 2. AI 두뇌 장착
    # 'best_model.pth'는 자네가 학습 코드에서 저장한 파일명이어야 하네
    ai_detector = NeuralCollisionDetector(model_path="best_model_v2.pth")

    # 결과 저장용 폴더
    vis_dir = "inference_result_comparison"
    os.makedirs(vis_dir, exist_ok=True)

    total_frames = 300
    
    for frame in tqdm(range(total_frames), desc="AI Inferencing"):
        # --- [A] 물리 시뮬레이션 스텝 ---
        sim.step() 
        
        # --- [B] 데이터 추출 ---
        # AI에게 먹여줄 데이터 (Input Feature)
        vel = sim.get_velocities() # (N, 3)
        pos = sim.get_positions()  # (N, 3) 
        
        # 비교를 위한 정답 데이터 (Ground Truth)
        real_penetration = sim.get_penetration_depth() # (N,)
        
        # --- [C] AI 추론 (The Moment of Truth) ---
        # 물리 엔진을 돌리는 대신, 순식간에 예측값을 받아오네
        pred_probability = ai_detector.predict(vel, pos, sim) # (N,) 0~1 사이 확률값
        
        # --- [D] 시각화 및 비교 저장 ---
        if frame % 10 == 0:
            # 1. AI가 예측한 결과를 OBJ로 저장 (빨간색 = AI가 충돌이라고 생각함)
            # 확률이므로 0.5를 기준으로 삼거나, 값 자체를 heatmap으로 씀
            save_inference_obj(
                f"{vis_dir}/ai_pred_{frame:03d}.obj",
                pos, pred_probability, width, height,
                mode="probability"
            )
            
            # 2. (선택사항) 실제 물리 엔진의 값도 저장해서 비교 (Ground Truth)
            # save_inference_obj(
            #    f"{vis_dir}/ground_truth_{frame:03d}.obj",
            #    pos, real_penetration, width, height, 
            #    mode="depth", thickness=sim.thickness
            # )

    print(f"✅ Inference Check Complete! Check '{vis_dir}' folder.")

# 시각화 함수 업데이트 (AI 확률용 모드 추가)
def save_inference_obj(filename, vertices, values, width, height, mode="depth", thickness=0.01):
    with open(filename, 'w') as f:
        f.write(f"# Visualization Mode: {mode}\n")
        
        for i, v in enumerate(vertices):
            val = values[i]
            r, g, b = 0.8, 0.8, 0.8 # 기본 회색
            
            if mode == "probability":
                # AI 확률 (0~1): 0이면 흰색, 1이면 빨간색
                # val은 0.0 ~ 1.0 사이
                r = 1.0
                g = 1.0 - val # 확률 높을수록 G, B 감소 -> 빨강
                b = 1.0 - val
                
            elif mode == "depth":
                # 기존 침투 깊이 로직 (자네 코드 재사용)
                diameter = thickness * 2.0
                ratio = (val - (diameter * 0.05)) / ((diameter * 0.3) - (diameter * 0.05))
                ratio = min(max(ratio, 0.0), 1.0)
                r, g, b = 1.0, 1.0 - ratio, 1.0 - ratio
            
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {r:.4f} {g:.4f} {b:.4f}\n")

        # Face 정보 (기존과 동일)
        for y in range(height - 1):
            for x in range(width - 1):
                idx = y * width + x + 1
                f.write(f"f {idx} {idx + width} {idx + 1}\n")
                f.write(f"f {idx + 1} {idx + width} {idx + width + 1}\n")

if __name__ == "__main__":
    # main_data_collection() # 이건 이제 주석 처리
    main_inference_test()    # 이걸 실행하게