import numpy as np
from numba import cuda
import math

# 상수는 기존과 동일하게 유지
HASH_TABLE_SIZE = 1000003
CELL_SIZE = 0.1

# ------------------------------------------------------------------------------
# [NEW] 1. Feature Extraction Kernel (GPU Only)
# ------------------------------------------------------------------------------
@cuda.jit
def compute_features_kernel(pos, vel, out_features, width, height, spacing):
    """
    [GPU Feature Extraction]
    AI 모델에 들어갈 Input Feature (vx, vy, vz, strain)를 GPU에서 즉시 계산합니다.
    CPU <-> GPU 데이터 전송 병목을 없애기 위함입니다.
    
    Args:
        pos: (N, 3) 현재 위치
        vel: (N, 3) 현재 속도
        out_features: (N, 4) 결과가 담길 버퍼 [vx, vy, vz, strain]
        width, height: 시뮬레이션 그리드 해상도
        spacing: Rest Length 기준값
    """
    idx = cuda.grid(1)
    num_particles = width * height
    
    if idx >= num_particles:
        return

    # 1. 속도 정보 복사 (Input 0~2)
    out_features[idx, 0] = vel[idx, 0]
    out_features[idx, 1] = vel[idx, 1]
    out_features[idx, 2] = vel[idx, 2]

    # 2. 기하 정보(Strain/Compression) 계산 (Input 3)
    # 1D Index -> 2D Grid Coordinate 변환
    x = idx % width
    y = idx // width
    
    current_strain = 0.0
    count = 0.0
    
    # (A) 오른쪽 이웃 (Right Neighbor)과의 거리 비율
    if x < width - 1:
        idx_right = idx + 1
        dx = pos[idx, 0] - pos[idx_right, 0]
        dy = pos[idx, 1] - pos[idx_right, 1]
        dz = pos[idx, 2] - pos[idx_right, 2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Strain = 현재길이 / 원래길이 (1.0보다 작으면 압축됨)
        current_strain += dist / spacing
        count += 1.0
        
    # (B) 아래쪽 이웃 (Bottom Neighbor)과의 거리 비율
    if y < height - 1:
        idx_bottom = idx + width
        dx = pos[idx, 0] - pos[idx_bottom, 0]
        dy = pos[idx, 1] - pos[idx_bottom, 1]
        dz = pos[idx, 2] - pos[idx_bottom, 2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        current_strain += dist / spacing
        count += 1.0

    # 평균 Strain 계산
    # (경계면에 있는 파티클은 연결이 적으므로 count로 나눔)
    if count > 0:
        val = current_strain / count
    else:
        val = 1.0 # 고립된 점은 변형 없음으로 간주
        
    out_features[idx, 3] = val


# ------------------------------------------------------------------------------
# [NEW] 2. Masked Self-Collision Kernel (The Optimization Core)
# ------------------------------------------------------------------------------
@cuda.jit
def solve_self_collision_masked_kernel(
    pos_pred, mass_inv, 
    cell_start, cell_end, 
    sorted_indices, particle_hashes, 
    risk_mask,       # [핵심] AI가 예측한 충돌 위험 마스크 (N, )
    num_particles, thickness, out_penetration
):
    """
    AI Mask 정보를 활용하여 안전한 파티클은 계산을 건너뛰는(Early Exit) 커널입니다.
    """
    idx = cuda.grid(1)
    if idx >= num_particles:
        return

    # --------------------------------------------------------------------------
    # [OPTIMIZATION] AI based Culling (Early Exit)
    # --------------------------------------------------------------------------
    # AI가 뱉은 확률값이 0.5 미만이면 "안전"하다고 판단하고 즉시 종료합니다.
    # GPU Warp Divergence가 발생하더라도, '안전한 영역'이 뭉쳐있다면(Coherent)
    # 해당 Warp 전체가 쉬게 되어 비약적인 성능 향상이 일어납니다.
    if risk_mask[idx] < 0.5:
        # 안전한 파티클은 침투 깊이도 0으로 기록
        out_penetration[idx] = 0.0 
        return 
    # --------------------------------------------------------------------------

    # 여기서부터는 기존 solve_self_collision_kernel 로직과 100% 동일합니다.
    # (위험하다고 판단된 파티클만 이 아래 코드를 실행합니다)
    
    w_i = mass_inv[idx]
    if w_i == 0.0: return
    
    px = pos_pred[idx, 0]
    py = pos_pred[idx, 1]
    pz = pos_pred[idx, 2]
    
    grid_x = int(math.floor(px / CELL_SIZE))
    grid_y = int(math.floor(py / CELL_SIZE))
    grid_z = int(math.floor(pz / CELL_SIZE))
    
    collision_stiffness = 0.2 
    max_displacement = thickness * 0.5 
    max_depth = 0.0 # 기록용

    # 3x3x3 Neighbor Search
    for z in range(-1, 2):
        for y in range(-1, 2):
            for x in range(-1, 2):
                neighbor_x = grid_x + x
                neighbor_y = grid_y + y
                neighbor_z = grid_z + z
                
                h = (neighbor_x * 73856093) ^ (neighbor_y * 19349663) ^ (neighbor_z * 83492791)
                h = h % HASH_TABLE_SIZE

                start_idx = cell_start[h]
                end_idx = cell_end[h]
                
                if start_idx == -1: continue 

                for k in range(start_idx, end_idx):
                    j = sorted_indices[k]
                    
                    # 자기 자신과의 충돌 방지
                    if idx == j: continue 
                    
                    jx = pos_pred[j, 0]
                    jy = pos_pred[j, 1]
                    jz = pos_pred[j, 2]
                    
                    dx = px - jx
                    dy = py - jy
                    dz = pz - jz
                    
                    dist_sq = dx*dx + dy*dy + dz*dz
                    min_dist = thickness * 2.0
                    
                    # 충돌 감지 (Distance Check)
                    if dist_sq < (min_dist * min_dist) and dist_sq > 1e-10:
                        dist = math.sqrt(dist_sq)
                        penetration = min_dist - dist
                        
                        if penetration > max_depth:
                            max_depth = penetration
                        
                        nx = dx / dist
                        ny = dy / dist
                        nz = dz / dist
                        
                        w_j = mass_inv[j]
                        w_sum = w_i + w_j
                        
                        # 위치 수정 (Position Correction)
                        if w_sum > 0:
                            s = (penetration / w_sum) * collision_stiffness
                            if s > max_displacement:
                                s = max_displacement

                            cuda.atomic.add(pos_pred, (idx, 0), nx * s * w_i)
                            cuda.atomic.add(pos_pred, (idx, 1), ny * s * w_i)
                            cuda.atomic.add(pos_pred, (idx, 2), nz * s * w_i)

    # 디버깅 및 시각화를 위해, 실제로 연산이 수행된 파티클의 침투 깊이를 기록
    out_penetration[idx] = max_depth

@cuda.jit
def solve_ground_collision_kernel(pos_pred, pos_old, vel, num_particles, ground_y, friction):
    """
    바닥 충돌 처리 및 마찰력 적용
    """
    idx = cuda.grid(1)
    if idx >= num_particles:
        return

    # 예측된 위치의 Y값이 바닥보다 아래라면?
    if pos_pred[idx, 1] < ground_y:
        # 1. 바닥 위로 강제 이동 (Hard Constraint)
        pos_pred[idx, 1] = ground_y
        
        # 2. 마찰력 적용 (수평 이동 저항)
        # 바닥에 닿은 순간, 미끄러지는 것을 방지하기 위해 
        # 이전 위치(pos_old) 쪽으로 살짝 되돌리거나 속도를 줄임
        
        # 간단한 방식: 속도 감쇠가 아니라 PBD에서는 위치를 수정함
        # 현재 스텝의 이동량(Delta)을 줄임
        
        # X축 마찰
        dx = pos_pred[idx, 0] - pos_old[idx, 0]
        pos_pred[idx, 0] = pos_old[idx, 0] + dx * (1.0 - friction)
        
        # Z축 마찰
        dz = pos_pred[idx, 2] - pos_old[idx, 2]
        pos_pred[idx, 2] = pos_old[idx, 2] + dz * (1.0 - friction)
        
        # (옵션) 수직 속도를 0으로 죽여서 튀어오름 방지
        vel[idx, 1] = 0.0