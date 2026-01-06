import os

import numpy as np
from Cloth.cloth import ClothSimulator
from tqdm import tqdm

# [기존 함수 유지] 색상(RGB)을 포함하여 OBJ 저장 & 침투 깊이 기반 보정
def save_obj_with_heatmap(filename, vertices, penetrations, width, height, thickness):
    """
    [Upgrade] Heatmap Color + UV Coordinates (Texture Mapping)
    """
    diameter = thickness * 1.5
    ignore_threshold = diameter * 0.05 
    critical_threshold = diameter * 0.3

    with open(filename, 'w') as f:
        f.write("# Powerful Cloth Sim with UVs\n")
        
        # 1. Vertices (v x y z r g b) - 히트맵 컬러 포함
        for i, v in enumerate(vertices):
            depth = penetrations[i]
            
            ratio = 0.0
            if depth > ignore_threshold:
                ratio = (depth - ignore_threshold) / (critical_threshold - ignore_threshold)
                ratio = min(max(ratio, 0.0), 1.0)
            
            r, g, b = 1.0, 1.0 - ratio, 1.0 - ratio
            # Blender는 OBJ의 Vertex Color를 지원함 (속성에서 확인 가능)
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {r:.4f} {g:.4f} {b:.4f}\n")

        # 2. UV Coordinates (vt u v) - [NEW] 텍스처 좌표 생성
        # 격자 형태이므로 0~1 사이 값으로 정규화하여 생성
        for y in range(height):
            for x in range(width):
                u = x / (width - 1)
                v = y / (height - 1)
                f.write(f"vt {u:.4f} {v:.4f}\n")

        # 3. Faces (f v1/vt1 v2/vt2 v3/vt3) - [NEW] 좌표 인덱스 연결
        for y in range(height - 1):
            for x in range(width - 1):
                # OBJ는 인덱스가 1부터 시작함
                # 현재 버텍스 순서와 UV 순서가 동일하게 생성되었으므로 인덱스를 같이 씀
                
                # Quad를 두 개의 Triangle로 나눔
                # (x, y), (x+1, y), (x, y+1), (x+1, y+1)
                
                idx_bl = (y * width + x) + 1       # Bottom-Left
                idx_br = (y * width + x + 1) + 1   # Bottom-Right
                idx_tl = ((y + 1) * width + x) + 1 # Top-Left
                idx_tr = ((y + 1) * width + x + 1) + 1 # Top-Right
                
                # Triangle 1 (BL - BR - TR) -> 반시계 방향 주의
                # f v/vt v/vt v/vt
                f.write(f"f {idx_bl}/{idx_bl} {idx_br}/{idx_br} {idx_tr}/{idx_tr}\n")
                
                # Triangle 2 (BL - TR - TL)
                f.write(f"f {idx_bl}/{idx_bl} {idx_tr}/{idx_tr} {idx_tl}/{idx_tl}\n")

def main_data_collection():
    print("🎓 Initialize Simulation for Ground Truth Collection...")

    width, height = 128, 128    # Resolution (128x128 정도면 학습용으로 적절합니다)
    sim = ClothSimulator(width, height, spacing=0.1)

    # 1. 학습 데이터셋 저장 폴더 (NPZ)
    dataset_dir = "dataset_flag_128"
    os.makedirs(dataset_dir, exist_ok=True)

    # 2. 시각화 확인용 폴더 (OBJ)
    vis_dir = "output_flag"
    os.makedirs(vis_dir, exist_ok=True)

    total_frames = 2000 # 충분한 데이터 확보를 위해 2000 프레임 권장
    print(f"Start simulation for {total_frames} frames...")

    for frame in tqdm(range(total_frames), desc="Collecting Data"):
        sim.step()

        # ---------------------------------------------------------
        # [중요] GPU -> CPU 데이터 가져오기
        # ---------------------------------------------------------
        # ClothSimulator 클래스에 get_velocities()가 구현되어 있어야 합니다.
        # (만약 없다면 d_vel.copy_to_host()를 리턴하는 함수를 추가하세요)
        
        pos = sim.get_positions()           # (N, 3) : 위치
        vel = sim.get_velocities()          # (N, 3) : 속도 [Input Feature]
        
        # 이름이 get_penetration_depths()인지 get_penetration_depth()인지 확인 필요
        # (이전 코드 맥락상 get_penetration_depths 일 가능성이 높음)
        penetration = sim.get_penetration_depth() # (N,) : 정답 라벨 [Ground Truth]

        # 기하학적 특성 추출
        geo_feature = sim.get_compression_feature(pos) # (N, 1)

        # ---------------------------------------------------------
        # [A] AI 학습용 데이터 저장 (.npz) - 매 프레임 저장 권장
        # ---------------------------------------------------------
        # 움직임의 연속성을 학습하려면 매 프레임 저장하는 것이 좋습니다.
        save_path = os.path.join(dataset_dir, f"data_{frame:04d}.npz")
        
        np.savez_compressed(
            save_path,
            pos=pos,    # 나중에 곡률(Curvature) 계산용
            vel=vel,    # 입력 피처 (속도가 빠르면 충돌 위험 Up)
            geo=geo_feature, # 기하학적 특성
            label=penetration # 정답 (0보다 크면 충돌 지역)
        )
        # ---------------------------------------------------------
        # [B] 시각화용 OBJ 저장 (10프레임마다) - 눈으로 확인용
        # ---------------------------------------------------------
        if frame % 10 == 0:
            save_obj_with_heatmap(
                f"{vis_dir}/cloth_{frame:03d}.obj",
                pos,
                penetration,
                width, height,
                sim.thickness
            )

    print(f"✅ Data Collection Finished! Saved to {dataset_dir}/")

if __name__ == "__main__":
    main_data_collection()