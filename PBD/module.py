import numpy as np
from numba import cuda
import math

# --- CUDA Kernels ---

@cuda.jit
def predict_position_kernel(pos, vel, pos_pred, mass_inv, dt, gravity_y, num_particles):
    """
    Step 1: 외력(중력)을 적용하고 미래 위치를 예측 (Explicit Integration)
    """
    idx = cuda.grid(1)
    if idx < num_particles:
        if mass_inv[idx] == 0.0: # 고정된 점(Fixed point)은 움직이지 않음
            pos_pred[idx, 0] = pos[idx, 0]
            pos_pred[idx, 1] = pos[idx, 1]
            pos_pred[idx, 2] = pos[idx, 2]
            return

        # v = v + g * dt
        vel[idx, 1] += gravity_y * dt

        # p* = p + v * dt
        pos_pred[idx, 0] = pos[idx, 0] + vel[idx, 0] * dt
        pos_pred[idx, 1] = pos[idx, 1] + vel[idx, 1] * dt
        pos_pred[idx, 2] = pos[idx, 2] + vel[idx, 2] * dt

@cuda.jit
def solve_distance_constraint_kernel(pos_pred, mass_inv, constraints, rest_lengths, compliance, dt, num_constraints):
    """
    Step 2: 거리 제약 조건 해결 (Distance Constraint Projection)
    XPBD 스타일로 Compliance(유연성)를 추가할 수도 있음.
    """
    idx = cuda.grid(1)
    if idx < num_constraints:
        id1 = constraints[idx, 0]
        id2 = constraints[idx, 1]
        
        w1 = mass_inv[id1]
        w2 = mass_inv[id2]
        w_sum = w1 + w2
        
        if w_sum == 0.0:
            return

        # 두 점 사이의 벡터와 거리 계산
        dx = pos_pred[id1, 0] - pos_pred[id2, 0]
        dy = pos_pred[id1, 1] - pos_pred[id2, 1]
        dz = pos_pred[id1, 2] - pos_pred[id2, 2]
        
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if dist == 0.0:
            return # 0으로 나누기 방지

        # 제약 조건 C(x) = |x1 - x2| - rest_length
        rest_len = rest_lengths[idx]
        correction = (dist - rest_len) / w_sum
        
        # XPBD의 경우 alpha = compliance / dt^2 추가 가능하지만 여기선 Hard constraint(PBD)로 진행
        
        grad_x = dx / dist
        grad_y = dy / dist
        grad_z = dz / dist
        
        # 위치 보정값 적용 (Atomic add를 써야 Race condition 방지 가능)
        # 하지만 여기서는 단순화를 위해 직접 더함 (Jacobi 방식이나 Graph Coloring 필요하지만 일단 Naive하게)
        
        # p1 보정 (-w1 * correction * gradient)
        cuda.atomic.add(pos_pred, (id1, 0), -w1 * correction * grad_x)
        cuda.atomic.add(pos_pred, (id1, 1), -w1 * correction * grad_y)
        cuda.atomic.add(pos_pred, (id1, 2), -w1 * correction * grad_z)
        
        # p2 보정 (+w2 * correction * gradient)
        cuda.atomic.add(pos_pred, (id2, 0), +w2 * correction * grad_x)
        cuda.atomic.add(pos_pred, (id2, 1), +w2 * correction * grad_y)
        cuda.atomic.add(pos_pred, (id2, 2), +w2 * correction * grad_z)

@cuda.jit
def update_velocity_kernel(pos, vel, pos_pred, dt, num_particles):
    """
    Step 3: 속도 갱신 + [NEW] 댐핑(Damping) 적용
    """
    idx = cuda.grid(1)
    if idx < num_particles:
        # PBD 속도 갱신
        v_new_x = (pos_pred[idx, 0] - pos[idx, 0]) / dt
        v_new_y = (pos_pred[idx, 1] - pos[idx, 1]) / dt
        v_new_z = (pos_pred[idx, 2] - pos[idx, 2]) / dt
        
        # [핵심 수정] Global Damping (공기 저항)
        # 속도를 매 프레임 아주 조금씩 줄여서 진동을 잡음
        # 0.99 ~ 0.995 정도가 적당함. (0.9는 너무 뻑뻑함)
        damping = 0.995 
        
        vel[idx, 0] = v_new_x * damping
        vel[idx, 1] = v_new_y * damping
        vel[idx, 2] = v_new_z * damping
        
        # 위치 확정
        pos[idx, 0] = pos_pred[idx, 0]
        pos[idx, 1] = pos_pred[idx, 1]
        pos[idx, 2] = pos_pred[idx, 2]

@cuda.jit
def solve_distance_constraint_colored_kernel(pos_pred, mass_inv, constraints, rest_lengths, 
                                             batch_indices, dt, k_stiffness):
    """
    [Updated] Colored Batch Solver
    - batch_indices: 이번 턴에 처리할 제약조건들의 인덱스 리스트
    - Atomic 연산 제거됨 (Race Condition 없음 보장)
    """
    tid = cuda.grid(1)
    
    # 이번 배치의 크기만큼만 스레드 실행
    if tid < batch_indices.shape[0]:
        # 간접 참조 (Indirect Addressing): 처리할 제약조건 번호를 가져옴
        c_idx = batch_indices[tid]
        
        id1 = constraints[c_idx, 0]
        id2 = constraints[c_idx, 1]
        
        w1 = mass_inv[id1]
        w2 = mass_inv[id2]
        w_sum = w1 + w2
        
        if w_sum == 0.0:
            return

        # P1, P2 위치 가져오기
        p1_x, p1_y, p1_z = pos_pred[id1, 0], pos_pred[id1, 1], pos_pred[id1, 2]
        p2_x, p2_y, p2_z = pos_pred[id2, 0], pos_pred[id2, 1], pos_pred[id2, 2]

        dx = p1_x - p2_x
        dy = p1_y - p2_y
        dz = p1_z - p2_z
        
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist == 0.0: return

        rest_len = rest_lengths[c_idx]
        correction = (dist - rest_len) / w_sum
        
        # PBD 정석 공식: Constraint Projection
        # Stiffness k는 이제 폭발 방지가 아니라, 재질의 특성(얼마나 잘 늘어나는가)을 제어함
        # k=1.0이면 완전 비신축성(Inextensible), k<1.0이면 고무줄
        
        grad_x = dx / dist
        grad_y = dy / dist
        grad_z = dz / dist
        
        # [핵심] Atomic 없이 직접 갱신! (Graph Coloring 덕분)
        # P1 Update
        pos_pred[id1, 0] -= w1 * correction * grad_x * k_stiffness
        pos_pred[id1, 1] -= w1 * correction * grad_y * k_stiffness
        pos_pred[id1, 2] -= w1 * correction * grad_z * k_stiffness
        
        # P2 Update
        pos_pred[id2, 0] += w2 * correction * grad_x * k_stiffness
        pos_pred[id2, 1] += w2 * correction * grad_y * k_stiffness
        pos_pred[id2, 2] += w2 * correction * grad_z * k_stiffness


# --- Spatial Hash Constants ---
HASH_TABLE_SIZE = 1000003  # 해시 테이블 크기 (충분히 크게)
CELL_SIZE = 0.1          # 격자 크기 (파티클 간격과 비슷하거나 약간 크게)

@cuda.jit
def compute_hash_kernel(pos, particle_hashes, particle_indices, num_particles):
    """
    각 파티클이 속한 Grid Cell의 Hash 값을 계산
    """
    idx = cuda.grid(1)
    if idx < num_particles:
        # 위치 가져오기
        x = pos[idx, 0]
        y = pos[idx, 1]
        z = pos[idx, 2]
        
        # Grid 좌표 계산 (양수로 변환하여 처리)
        grid_x = int(math.floor(x / CELL_SIZE))
        grid_y = int(math.floor(y / CELL_SIZE))
        grid_z = int(math.floor(z / CELL_SIZE))
        
        # Spatial Hash Function (Large Primes)
        # (x * p1 ^ y * p2 ^ z * p3) % table_size
        h = (grid_x * 73856093) ^ (grid_y * 19349663) ^ (grid_z * 83492791)
        h = h % HASH_TABLE_SIZE
        
        particle_hashes[idx] = h
        particle_indices[idx] = idx

@cuda.jit
def find_cell_start_end_kernel(particle_hashes, cell_start, cell_end, num_particles):
    """
    정렬된 해시 배열을 보고, 각 Cell이 시작되는 인덱스와 끝나는 인덱스를 기록
    """
    idx = cuda.grid(1)
    if idx < num_particles:
        hash_val = particle_hashes[idx]
        
        # 첫 번째 요소 처리
        if idx == 0:
            cell_start[hash_val] = idx
        else:
            prev_hash = particle_hashes[idx - 1]
            if hash_val != prev_hash:
                cell_start[hash_val] = idx
                cell_end[prev_hash] = idx # 이전 셀의 끝
        
        # 마지막 요소 처리
        if idx == num_particles - 1:
            cell_end[hash_val] = idx + 1

@cuda.jit
def solve_self_collision_kernel(pos_pred, mass_inv, cell_start, cell_end, 
                                sorted_indices, particle_hashes, num_particles, 
                                thickness, penetration_buffer): # [NEW] 인자 추가됨
    """
    Self-Collision 해결 및 Penetration Depth 기록
    """
    idx = cuda.grid(1)
    if idx < num_particles:
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

        # [NEW] 이 파티클이 겪는 최대 침투 깊이를 추적하기 위한 변수
        max_depth = 0.0

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
                        if idx == j: continue 
                        
                        jx = pos_pred[j, 0]
                        jy = pos_pred[j, 1]
                        jz = pos_pred[j, 2]
                        
                        dx = px - jx
                        dy = py - jy
                        dz = pz - jz
                        
                        dist_sq = dx*dx + dy*dy + dz*dz
                        min_dist = thickness * 2.0
                        
                        if dist_sq < (min_dist * min_dist) and dist_sq > 1e-10:
                            dist = math.sqrt(dist_sq)
                            penetration = min_dist - dist
                            
                            # [NEW] 최대 침투 깊이 기록 (시각화/데이터 수집용)
                            if penetration > max_depth:
                                max_depth = penetration
                            
                            nx = dx / dist
                            ny = dy / dist
                            nz = dz / dist
                            
                            w_j = mass_inv[j]
                            w_sum = w_i + w_j
                            
                            if w_sum > 0:
                                s = (penetration / w_sum) * collision_stiffness
                                if s > max_displacement:
                                    s = max_displacement

                                cuda.atomic.add(pos_pred, (idx, 0), nx * s * w_i)
                                cuda.atomic.add(pos_pred, (idx, 1), ny * s * w_i)
                                cuda.atomic.add(pos_pred, (idx, 2), nz * s * w_i)

        # [NEW] 루프가 끝난 후, 이 파티클의 최대 침투 깊이를 버퍼에 저장
        # (다음 프레임 AI 학습 데이터의 라벨로 사용됨)
        penetration_buffer[idx] = max_depth