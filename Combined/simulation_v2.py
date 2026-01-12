import torch
import torch.nn as nn
import numpy as np
from numba import cuda
import sys
import os
import math
import argparse
import csv
from tqdm import tqdm

# 경로 설정
sys.path.append('../')

THRESHOLD = 0.01


# ==============================================================================
# [Visualizer] Heatmap Saver
# ==============================================================================
def save_obj_with_heatmap(filename, vertices, values, width, height, thickness):
    """
    values: 0.0 ~ 1.0 사이의 값 (Risk Mask 또는 Penetration)
    빨간색이 잘 보이도록 임계값을 조정했습니다.
    """
    diameter = thickness * 1.5
    # 시각화 민감도 설정
    ignore_threshold = THRESHOLD  # 0.01 이상이면 색칠 시작
    critical_threshold = 0.02 # 0.6 이상이면 새빨간색

    with open(filename, 'w') as f:
        f.write("# Powerful Cloth Sim with Heatmap\n")
        
        # 1. Vertices & Colors
        for i, v in enumerate(vertices):
            val = values[i]
            
            ratio = 0.0
            if val > ignore_threshold:
                ratio = (val - ignore_threshold) / (critical_threshold - ignore_threshold)
                ratio = min(max(ratio, 0.0), 1.0)
            
            # White(Safe) -> Red(Risk)
            r = 1.0
            g = 1.0 - ratio
            b = 1.0 - ratio
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {r:.4f} {g:.4f} {b:.4f}\n")

        # 2. UVs
        for y in range(height):
            for x in range(width):
                u = x / (width - 1)
                v = y / (height - 1)
                f.write(f"vt {u:.4f} {v:.4f}\n")

        # 3. Faces
        for y in range(height - 1):
            for x in range(width - 1):
                idx_bl = (y * width + x) + 1       
                idx_br = (y * width + x + 1) + 1   
                idx_tl = ((y + 1) * width + x) + 1 
                idx_tr = ((y + 1) * width + x + 1) + 1 
                f.write(f"f {idx_bl}/{idx_bl} {idx_br}/{idx_br} {idx_tr}/{idx_tr}\n")
                f.write(f"f {idx_bl}/{idx_bl} {idx_tr}/{idx_tr} {idx_tl}/{idx_tl}\n")

# ==============================================================================
# [Kernels] CUDA Physics Kernels
# ==============================================================================
from cloth_cuda_optimized import compute_features_kernel, solve_self_collision_masked_kernel, solve_ground_collision_kernel
from PBD.module import predict_position_kernel, compute_hash_kernel, find_cell_start_end_kernel
from PBD.coloring import compute_graph_coloring
from PBD.module import solve_distance_constraint_colored_kernel 

# [NEW] Hysteresis Update Kernel (시간적 이력 관리)
@cuda.jit
def update_hysteresis_kernel(risk_mask, active_timer, decay_frames, num_particles):
    i = cuda.grid(1)
    if i < num_particles:
        # 1. AI가 위험하다고 판단(Active)했으면 타이머 리셋 (예: 15프레임)
        if risk_mask[i] > THRESHOLD: # Threshold
            active_timer[i] = 15 
        
        # 2. 타이머 관리
        if active_timer[i] > 0:
            active_timer[i] -= 1
            risk_mask[i] = 1.0 # 타이머가 살아있으면 강제로 Active 유지
        else:
            risk_mask[i] = 0.0 # 타이머 종료 시 비활성화

# [NEW] Adaptive Damping Kernel (에너지 제어)
@cuda.jit
def update_velocity_adaptive_kernel(pos, vel, pos_pred, risk_mask, dt, num_particles):
    i = cuda.grid(1)
    if i < num_particles:
        # PBD Velocity Update: v = (x_new - x_old) / dt
        new_vx = (pos_pred[i, 0] - pos[i, 0]) / dt
        new_vy = (pos_pred[i, 1] - pos[i, 1]) / dt
        new_vz = (pos_pred[i, 2] - pos[i, 2]) / dt
        
        # [핵심] Adaptive Damping Logic
        # Active(위험) 상태: 0.90 (강한 댐핑으로 충돌 에너지 흡수)
        # Inactive(안전) 상태: 0.995 (약한 댐핑으로 자연스러운 움직임)
        damping = 0.995
        if risk_mask[i] > THRESHOLD:
            damping = 0.90 
            
        vel[i, 0] = new_vx * damping
        vel[i, 1] = new_vy * damping
        vel[i, 2] = new_vz * damping
        
        # 위치 확정 Update
        pos[i, 0] = pos_pred[i, 0]
        pos[i, 1] = pos_pred[i, 1]
        pos[i, 2] = pos_pred[i, 2]

# ==============================================================================
# [AI Model] Pruning Model Architecture
# ==============================================================================
class CollisionPruningModel(nn.Module):
    def __init__(self):
        super(CollisionPruningModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid() 
        )
    def forward(self, x): return self.net(x)

# ==============================================================================
# [Simulation Engine] Main Class
# ==============================================================================
class PowerfulClothSim:
    def __init__(self, width, height, model_path, spacing=0.1):
        self.width, self.height = width, height
        self.num_particles = width * height
        self.spacing = spacing
        
        # 물리 파라미터
        self.dt = 0.003
        self.substeps = 5
        self.gravity = -9.8

        print(f"⚡ Init PowerfulClothSim ({width}x{height}) [Stable Mode]")

        # 1. Host Data Setup (Flag Scene)
        pos_host = np.zeros((self.num_particles, 3), dtype=np.float32)
        start_y = 2.0 
        flag_wave_amplitude = spacing * 0.6  
        flag_wave_frequency = 2.0            

        for y in range(height):
            for x in range(width):
                idx = y * width + x
                pos_x = x * spacing
                pos_y = (height - y - 1) * spacing + start_y
                pos_z = math.sin(x * flag_wave_frequency * math.pi / width) * flag_wave_amplitude
                pos_z *= (1.0 - y / (height-1)) if height > 1 else 1.0
                pos_host[idx] = [pos_x, pos_y, pos_z]

        # 2. Constraints (Stress-Free Initialization)
        constraints = []
        rest_lengths_list = []
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                p1 = pos_host[idx]
                # Structural & Shear
                neighbors = []
                if x < width - 1: neighbors.append(idx + 1)
                if y < height - 1: neighbors.append(idx + width)
                if x < width - 1 and y < height - 1:
                    neighbors.append(idx + width + 1)
                    neighbors.append(idx + width) # For diagonal cross (needs correct pair)
                
                # Explicit additions for clarity
                if x < width - 1: # Right
                    idx_n = idx + 1
                    constraints.append([idx, idx_n])
                    rest_lengths_list.append(np.linalg.norm(p1 - pos_host[idx_n]))
                if y < height - 1: # Down
                    idx_n = idx + width
                    constraints.append([idx, idx_n])
                    rest_lengths_list.append(np.linalg.norm(p1 - pos_host[idx_n]))
                if x < width - 1 and y < height - 1: # Diagonals
                    # ↘
                    idx_br = idx + width + 1
                    constraints.append([idx, idx_br])
                    rest_lengths_list.append(np.linalg.norm(p1 - pos_host[idx_br]))
                    # ↙
                    idx_tr = idx + 1
                    idx_bl = idx + width
                    constraints.append([idx_tr, idx_bl])
                    rest_lengths_list.append(np.linalg.norm(pos_host[idx_tr] - pos_host[idx_bl]))

        # 3. GPU Memory Allocation
        print("🎨 Computing Graph Coloring...")
        color_batches_host = compute_graph_coloring(self.num_particles, constraints)
        self.d_color_batches = [cuda.to_device(batch) for batch in color_batches_host]
        
        self.d_pos = cuda.to_device(pos_host)
        self.d_pos_pred = cuda.device_array_like(self.d_pos)
        self.d_vel = cuda.to_device(np.zeros_like(pos_host))
        self.d_constraints = cuda.to_device(np.array(constraints, dtype=np.int32))
        self.d_rest_lengths = cuda.to_device(np.array(rest_lengths_list, dtype=np.float32))
        
        mass_inv = np.ones(self.num_particles, dtype=np.float32)
        mass_inv[0] = 0.0 # Pinning one corner
        self.d_mass_inv = cuda.to_device(mass_inv)
        
        # Spatial Hashing
        self.HASH_SIZE = 2999999 
        self.d_particle_hashes = cuda.device_array(self.num_particles, dtype=np.int32)
        self.d_particle_indices = cuda.device_array(self.num_particles, dtype=np.int32)
        self.d_cell_start = cuda.device_array(self.HASH_SIZE, dtype=np.int32)
        self.d_cell_end = cuda.device_array(self.HASH_SIZE, dtype=np.int32)
        self.thickness = spacing * 0.3
        self.d_penetration = cuda.device_array(self.num_particles, dtype=np.float32)

        # [NEW] Hysteresis Timer Buffer
        self.d_active_timer = cuda.device_array(self.num_particles, dtype=np.int32)
        self.d_active_timer[:] = 0

        self.threads = 256
        self.blocks = (self.num_particles + 255) // 256
        
        # 4. AI Setup
        print(f"🧠 Loading AI Brain from {model_path}...")
        self.ai_model = CollisionPruningModel().cuda()
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cuda')
            # Handle possible key mismatch
            try:
                self.ai_model.load_state_dict(checkpoint, strict=False)
            except:
                clean_state = {k.replace("net.", ""): v for k, v in checkpoint.items()}
                self.ai_model.net.load_state_dict(clean_state, strict=False)
        else:
            print("⚠️ Model file not found. Initializing random weights (for testing only).")

        self.ai_model.eval()
        
        # Safe compile for Windows
        try:
            if hasattr(torch, "compile") and os.name != 'nt':
                self.ai_model = torch.compile(self.ai_model)
                print("🚀 PyTorch 2.x Compiled Model Activated!")
        except: pass

        self.d_features = cuda.device_array((self.num_particles, 4), dtype=np.float32)
        self.d_risk_mask = cuda.device_array(self.num_particles, dtype=np.float32)
        self.d_risk_mask[:] = 0.0

        self.frame_count = 0
        self.ai_interval = 5 # 검사 주기 5프레임

        print("✅ Simulation Engine Ready.")

    def _numba_to_torch(self, arr): return torch.as_tensor(arr, device='cuda')

    def _run_ai_culling(self):
        # 1. Features
        compute_features_kernel[self.blocks, self.threads](
            self.d_pos, self.d_vel, self.d_features, self.width, self.height, self.spacing
        )
        
        # 2. Inference
        input_tensor = self._numba_to_torch(self.d_features)
        
        # [수정] 학습 때 사용한 통계치로 정규화 (Hardcoded Stats)
        # Vel: (x - (-0.097)) / 0.458
        input_tensor[:, :3] = (input_tensor[:, :3] - (-0.097470)) / (0.458327 + 1e-6)
        
        # Geo: (x - 1.141) / 0.201
        input_tensor[:, 3] = (input_tensor[:, 3] - 1.141827) / (0.201266 + 1e-6)

        with torch.no_grad():            
            probs = self.ai_model(input_tensor)
            # AI 판단 (0.2 이상이면 위험) -> 임시 마스크에 저장
            mask_tensor = (probs > THRESHOLD).float().squeeze() 
            # *주의: 여기서 바로 d_risk_mask에 덮어쓰지 않고, Hysteresis 커널이 판단하도록 함*
            # 하지만 구현 편의상 일단 복사 후 커널에서 처리
            self.d_risk_mask[:] = cuda.as_cuda_array(mask_tensor.contiguous())

        return probs

    def _sort_particles_torch(self):
        hashes = self._numba_to_torch(self.d_particle_hashes)
        indices = self._numba_to_torch(self.d_particle_indices)
        sorted_idx = torch.argsort(hashes)
        self.d_particle_hashes[:] = cuda.as_cuda_array(hashes[sorted_idx].contiguous())
        self.d_particle_indices[:] = cuda.as_cuda_array(indices[sorted_idx].contiguous())

    def step(self):
        dt_sub = self.dt / self.substeps

        if self.frame_count < 20:
            self.d_risk_mask[:] = 0.0 # 초반엔 무조건 안전하다고 가정        
        elif self.frame_count % self.ai_interval == 0:
            probs = self._run_ai_culling()
            
            # [디버깅] 가끔 모델이 얼마나 겁먹었는지(평균 확률) 확인
            if self.frame_count % 100 == 0:
                # _run_ai_culling 안에서 probs를 찍어보거나 여기서 확인 가능
                print(probs)
        
        # Hysteresis 커널 실행 (Timer 갱신 및 마스크 유지)
        update_hysteresis_kernel[self.blocks, self.threads](
            self.d_risk_mask, self.d_active_timer, 1, self.num_particles
        )
        
        self.frame_count += 1
        # # Pin Release Logic
        # if self.frame_count == 500:
        #     mass_inv = self.d_mass_inv.copy_to_host()
        #     mass_inv[0] = 1.0
        #     self.d_mass_inv = cuda.to_device(mass_inv)
        #     self.dt = 0.01

        # [Phase 2] Physics Substeps
        for _ in range(self.substeps):
            predict_position_kernel[self.blocks, self.threads](
                self.d_pos, self.d_vel, self.d_pos_pred, self.d_mass_inv, dt_sub, self.gravity, self.num_particles
            )
            
            # Distance Constraints
            for batch in self.d_color_batches:
                solve_distance_constraint_colored_kernel[(batch.shape[0]+255)//256, 256](
                    self.d_pos_pred, self.d_mass_inv, self.d_constraints, self.d_rest_lengths, batch, dt_sub, 0.8
                )
            
            # Collision Detection Setup
            self.d_cell_start[:] = -1; self.d_cell_end[:] = -1
            compute_hash_kernel[self.blocks, self.threads](
                self.d_pos_pred, self.d_particle_hashes, self.d_particle_indices, self.num_particles
            )
            self._sort_particles_torch()
            find_cell_start_end_kernel[self.blocks, self.threads](
                self.d_particle_hashes, self.d_cell_start, self.d_cell_end, self.num_particles
            )

            # Ground Collision
            solve_ground_collision_kernel[self.blocks, self.threads](
                self.d_pos_pred, self.d_pos, self.d_vel, self.num_particles, 0.0, 0.7
            )
            
            # [Core] AI Masked Self-Collision
            solve_self_collision_masked_kernel[self.blocks, self.threads](
                self.d_pos_pred, self.d_mass_inv, self.d_cell_start, self.d_cell_end,
                self.d_particle_indices, self.d_particle_hashes, self.d_risk_mask,
                self.num_particles, self.thickness, self.d_penetration
            )
            
            # [Modified] Adaptive Damping Velocity Update
            update_velocity_adaptive_kernel[self.blocks, self.threads](
                self.d_pos, self.d_vel, self.d_pos_pred, self.d_risk_mask, dt_sub, self.num_particles
            )

    # Accessors
    def get_positions(self): return self.d_pos.copy_to_host()
    def get_risk_mask(self): return self.d_risk_mask.copy_to_host()
    def get_penetrations(self): return self.d_penetration.copy_to_host()

# ==============================================================================
# Main Logic
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=int, default=2, help="2: Extract OBJ")
    args = parser.parse_args()

    # 학습된 모델 경로 지정
    MODEL_PATH = "../MLP/best_model_norm.pth" 
    
    # 1. Init
    sim = PowerfulClothSim(128, 128, MODEL_PATH, spacing=0.1)
    
    # 2. Run
    OUTPUT_DIR = "extracted_objs_stable_v1"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    TOTAL_FRAMES = 4000
    
    print(f"🚀 Running Simulation... Saving to {OUTPUT_DIR}")
    
    for i in tqdm(range(TOTAL_FRAMES)):
        sim.step()
        
        if i % 10 == 0:
            pos = sim.get_positions()
            # [시각화 핵심] AI가 활성화한(Active) 마스크를 시각화합니다.
            # 이 값이 1.0(Red)인 곳은 물리 연산이 수행되고 댐핑이 강하게 걸린 곳입니다.
            risk_mask = sim.get_risk_mask()
            
            filename = os.path.join(OUTPUT_DIR, f"cloth_{i:04d}.obj")
            save_obj_with_heatmap(filename, pos, risk_mask, sim.width, sim.height, sim.thickness)

    print("\n✅ Done!")