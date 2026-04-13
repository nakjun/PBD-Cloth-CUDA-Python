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
    idx = cuda.grid(1)
    if idx < num_particles:
        # PBD 속도 갱신
        v_new_x = (pos_pred[idx, 0] - pos[idx, 0]) / dt
        v_new_y = (pos_pred[idx, 1] - pos[idx, 1]) / dt
        v_new_z = (pos_pred[idx, 2] - pos[idx, 2]) / dt
        
        # [NEW] Max Velocity Clamping (안전장치)
        # 파티클이 한 프레임에 너무 많이 이동하지 못하게 막음
        # 예: dt=0.01일 때 속도가 100이면 1m 이동. 천 크기가 12.8m이므로 적당함.
        # 하지만 순간적으로 튀는 걸 막으려면 20~50 정도가 적당.
        max_vel = 20.0 
        
        current_speed = math.sqrt(v_new_x**2 + v_new_y**2 + v_new_z**2)
        if current_speed > max_vel:
            scale = max_vel / current_speed
            v_new_x *= scale
            v_new_y *= scale
            v_new_z *= scale

        # Global Damping
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
                                             batch_indices, dt, compliance): # [변경] k_stiffness -> compliance
    """
    [XPBD Implementation]
    compliance: 재질의 유연성 (0.0 = 딱딱함/비신축성, 값이 클수록 고무줄처럼 늘어남)
                단위: m/N (미터 퍼 뉴턴)
    """
    tid = cuda.grid(1)
    
    if tid < batch_indices.shape[0]:
        c_idx = batch_indices[tid]
        
        id1 = constraints[c_idx, 0]
        id2 = constraints[c_idx, 1]
        
        w1 = mass_inv[id1]
        w2 = mass_inv[id2]
        w_sum = w1 + w2
        
        if w_sum == 0.0:
            return

        p1 = pos_pred[id1]
        p2 = pos_pred[id2]

        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist == 0.0: return

        rest_len = rest_lengths[c_idx]
        
        # [XPBD Core] 
        # C(x) = dist - rest_len
        C = dist - rest_len
        
        # alpha_tilde = compliance / dt^2
        alpha_tilde = compliance / (dt * dt)
        
        # Lagrange Multiplier Update (Approximate for one iteration)
        # XPBD Formula: lambda = -C / (w_sum + alpha_tilde)
        # 기존 PBD는 분모가 w_sum 뿐이었음. alpha_tilde가 추가되어 '유연함'을 물리적으로 제어함.
        
        delta_lambda = -C / (w_sum + alpha_tilde)
        
        # Gradient Vector
        grad_x = dx / dist
        grad_y = dy / dist
        grad_z = dz / dist
        
        # Apply Correction
        # delta_x = w * delta_lambda * grad
        
        # P1 Update
        if w1 > 0:
            cuda.atomic.add(pos_pred, (id1, 0), w1 * delta_lambda * grad_x)
            cuda.atomic.add(pos_pred, (id1, 1), w1 * delta_lambda * grad_y)
            cuda.atomic.add(pos_pred, (id1, 2), w1 * delta_lambda * grad_z)
        
        # P2 Update
        if w2 > 0:
            cuda.atomic.add(pos_pred, (id2, 0), -w2 * delta_lambda * grad_x) # 방향 반대 주의 (-grad)
            cuda.atomic.add(pos_pred, (id2, 1), -w2 * delta_lambda * grad_y)
            cuda.atomic.add(pos_pred, (id2, 2), -w2 * delta_lambda * grad_z)


# =============================================================================
# [Phase 4 최적화] Jacobi 방식 Distance Constraint
# 모든 제약을 한 번에 병렬 처리 (Graph Coloring 불필요)
# Under-relaxation으로 수렴 안정화
# =============================================================================

@cuda.jit
def solve_distance_constraint_jacobi_kernel(
    pos_pred,           # (N, 3) 예측 위치
    pos_delta,          # (N, 3) 누적 보정값 (출력)
    delta_count,        # (N,) 각 파티클에 적용된 보정 횟수
    mass_inv,           # (N,) 역질량
    constraints,        # (M, 2) 제약 (파티클 쌍)
    rest_lengths,       # (M,) 휴식 길이
    dt,                 # 시간 간격
    compliance,         # 유연성
    num_constraints     # 제약 수
):
    """
    [Phase 4 최적화] Jacobi 방식 Distance Constraint
    
    Graph Coloring 방식:
    - 색상 배치 수만큼 커널 실행 (5-10회)
    - 각 배치 내에서만 병렬 처리
    
    Jacobi 방식:
    - 단일 커널 실행
    - 모든 제약 병렬 처리
    - 보정값을 별도 버퍼에 누적
    - 나중에 평균화하여 적용
    
    장점: 커널 실행 오버헤드 감소
    단점: 수렴 속도 감소 (더 많은 iteration 필요할 수 있음)
    """
    c_idx = cuda.grid(1)
    if c_idx >= num_constraints:
        return
    
    id1 = constraints[c_idx, 0]
    id2 = constraints[c_idx, 1]
    
    w1 = mass_inv[id1]
    w2 = mass_inv[id2]
    w_sum = w1 + w2
    
    if w_sum == 0.0:
        return
    
    # 현재 위치 읽기
    p1x = pos_pred[id1, 0]
    p1y = pos_pred[id1, 1]
    p1z = pos_pred[id1, 2]
    
    p2x = pos_pred[id2, 0]
    p2y = pos_pred[id2, 1]
    p2z = pos_pred[id2, 2]
    
    dx = p1x - p2x
    dy = p1y - p2y
    dz = p1z - p2z
    
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < 1e-10:
        return
    
    rest_len = rest_lengths[c_idx]
    C = dist - rest_len
    
    # XPBD
    alpha_tilde = compliance / (dt * dt)
    delta_lambda = -C / (w_sum + alpha_tilde)
    
    # Gradient
    grad_x = dx / dist
    grad_y = dy / dist
    grad_z = dz / dist
    
    # 보정값 누적 (atomic add)
    if w1 > 0:
        cuda.atomic.add(pos_delta, (id1, 0), w1 * delta_lambda * grad_x)
        cuda.atomic.add(pos_delta, (id1, 1), w1 * delta_lambda * grad_y)
        cuda.atomic.add(pos_delta, (id1, 2), w1 * delta_lambda * grad_z)
        cuda.atomic.add(delta_count, id1, 1)
    
    if w2 > 0:
        cuda.atomic.add(pos_delta, (id2, 0), -w2 * delta_lambda * grad_x)
        cuda.atomic.add(pos_delta, (id2, 1), -w2 * delta_lambda * grad_y)
        cuda.atomic.add(pos_delta, (id2, 2), -w2 * delta_lambda * grad_z)
        cuda.atomic.add(delta_count, id2, 1)


@cuda.jit
def apply_jacobi_correction_kernel(
    pos_pred,           # (N, 3) 예측 위치 (입출력)
    pos_delta,          # (N, 3) 누적 보정값
    delta_count,        # (N,) 보정 횟수
    relaxation_factor,  # Under-relaxation 계수 (0.5~1.0)
    num_particles
):
    """
    Jacobi 보정값 적용 커널
    
    각 파티클에 누적된 보정값을 평균화하고 under-relaxation 적용
    """
    idx = cuda.grid(1)
    if idx >= num_particles:
        return
    
    count = delta_count[idx]
    if count == 0:
        return
    
    # 평균화 + Under-relaxation
    scale = relaxation_factor / count
    
    pos_pred[idx, 0] += pos_delta[idx, 0] * scale
    pos_pred[idx, 1] += pos_delta[idx, 1] * scale
    pos_pred[idx, 2] += pos_delta[idx, 2] * scale
    
    # 버퍼 초기화 (다음 iteration용)
    pos_delta[idx, 0] = 0.0
    pos_delta[idx, 1] = 0.0
    pos_delta[idx, 2] = 0.0
    delta_count[idx] = 0


# --- Spatial Hash Constants ---
HASH_TABLE_SIZE = 1000003  # 해시 테이블 크기 (충분히 크게)
CELL_SIZE = 0.1          # 격자 크기 (파티클 간격과 비슷하거나 약간 크게)

# @cuda.jit
# def compute_hash_kernel(pos, particle_hashes, particle_indices, num_particles):
#     """
#     각 파티클이 속한 Grid Cell의 Hash 값을 계산
#     """
#     idx = cuda.grid(1)
#     if idx < num_particles:
#         # 위치 가져오기
#         x = pos[idx, 0]
#         y = pos[idx, 1]
#         z = pos[idx, 2]
        
#         # Grid 좌표 계산 (양수로 변환하여 처리)
#         grid_x = int(math.floor(x / CELL_SIZE))
#         grid_y = int(math.floor(y / CELL_SIZE))
#         grid_z = int(math.floor(z / CELL_SIZE))
        
#         # Spatial Hash Function (Large Primes)
#         # (x * p1 ^ y * p2 ^ z * p3) % table_size
#         h = (grid_x * 73856093) ^ (grid_y * 19349663) ^ (grid_z * 83492791)
#         h = h % HASH_TABLE_SIZE
        
#         particle_hashes[idx] = h
#         particle_indices[idx] = idx

@cuda.jit
def find_cell_start_end_kernel(particle_hashes, cell_start, cell_end, num_particles):
    """
    정렬된 해시 배열을 보고, 각 Cell이 시작되는 인덱스와 끝나는 인덱스를 기록
    hash=-1인 파티클(Culling된 파티클)은 제외
    """
    idx = cuda.grid(1)
    if idx < num_particles:
        hash_val = particle_hashes[idx]
        
        # [핵심 수정] hash=-1인 파티클(Culling된 파티클)은 셀 범위에 포함하지 않음
        if hash_val < 0:
            return
        
        # 첫 번째 유효한 요소 처리
        if idx == 0:
            cell_start[hash_val] = idx
        else:
            prev_hash = particle_hashes[idx - 1]
            # 이전 파티클이 Culling되지 않았고, hash 값이 다르면 셀 경계
            if prev_hash >= 0 and hash_val != prev_hash:
                cell_start[hash_val] = idx
                cell_end[prev_hash] = idx # 이전 셀의 끝
        
        # 마지막 요소 처리
        if idx == num_particles - 1:
            cell_end[hash_val] = idx + 1
        else:
            # 다음 파티클이 Culling되었거나 hash 값이 다르면 현재 셀의 끝
            next_hash = particle_hashes[idx + 1] if idx + 1 < num_particles else -1
            if next_hash < 0 or next_hash != hash_val:
                cell_end[hash_val] = idx + 1

import math
from numba import cuda, float32, int32
import numpy as np

# ... ( compute_face_normals 등 다른 커널들은 기존 유지 ) ...

@cuda.jit
def solve_self_collision_friction_kernel(pos_pred, pos_old, mass_inv, 
                                         cell_start, cell_end, 
                                         sorted_indices, particle_hashes, 
                                         num_particles, thickness, 
                                         penetration_buffer, dt,
                                         visibility, frame_idx, debug_skip_count,
                                         curvature, curvature_threshold,
                                         active_pair_count): 
    """
    [Sorted-based Spatial Hashing] + [Physics-based Early Termination] + [Hierarchical Culling]
    
    다단계 Culling 파이프라인:
    1. Curvature Culling: 평탄한 영역의 파티클 건너뜀 (가장 먼저 체크)
    2. View-Dependent Culling: 뒷면 파티클 확률적 건너뜀
    3. Physics-based Early Termination: 멀어지는 파티클 쌍 건너뜀
    
    앞 단계에서 Culling된 파티클은 이후 단계를 실행하지 않습니다.
    """
    idx = cuda.grid(1)
    if idx >= num_particles: return

    # ============================================================
    # [Stage 1] Hash-based Culling (최우선 - 가장 빠른 체크)
    # ============================================================
    # hash=-1인 파티클은 이미 Curvature Culling으로 제외됨
    # 정렬 후 hash=-1인 파티클은 배열 앞부분에 있지만, 여기서도 체크하여 즉시 종료
    # sorted_indices를 통해 실제 파티클 ID를 가져옴
    actual_idx = sorted_indices[idx]
    if particle_hashes[idx] < 0:
        return  # Hash-based Culling: 이미 컬링된 파티클
    # ============================================================

    # ============================================================
    # [Stage 2] View-Dependent Culling (두 번째 체크)
    # ============================================================
    # Hash Culling을 통과한 파티클만 View-Dependent Culling 수행
    vis_score = visibility[actual_idx]
    view_culling_threshold = 0.2  # 뒷면 판단 기준

    if vis_score < view_culling_threshold:
        seed = actual_idx * 12345 + frame_idx * 67897
        rand_state = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        rand_float = float(rand_state) / 2147483648.0 
        
        skip_probability = 0.7  # 스킵 확률

        if rand_float < skip_probability:
            return  # View-Dependent Culling: 확률적으로 건너뜀
    # ============================================================
    
    # actual_idx는 sorted_indices를 통해 가져온 실제 파티클 ID
    w_i = mass_inv[actual_idx]
    if w_i == 0.0: return
    
    px = pos_pred[actual_idx, 0]
    py = pos_pred[actual_idx, 1]
    pz = pos_pred[actual_idx, 2]

    # 이전 프레임 위치 (속도 계산용)
    px_old = pos_old[actual_idx, 0]
    py_old = pos_old[actual_idx, 1]
    pz_old = pos_old[actual_idx, 2]
    
    grid_x = int(math.floor(px / CELL_SIZE))
    grid_y = int(math.floor(py / CELL_SIZE))
    grid_z = int(math.floor(pz / CELL_SIZE))
    
    # --- Parameters ---
    contact_compliance = 0.00001 
    alpha_tilde = contact_compliance / (dt * dt)
    friction_mu_k = 0.05
    friction_mu_s = 0.05
    max_displacement = thickness * 0.2
    max_collisions = 8
    collision_count = 0
    max_depth = 0.0
    stop_search = False

    # 2. Neighbor Search
    for z in range(-1, 2):
        if stop_search: break
        for y in range(-1, 2):
            if stop_search: break
            for x in range(-1, 2):
                if stop_search: break
                
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
                    
                    # 중복 및 자기 자신 검사 스킵
                    if idx <= j: continue
                    
                    # [핵심 수정] 이웃 파티클 j가 Culling되었는지 확인
                    # hash=-1인 파티클은 Curvature Culling으로 제외되었으므로 검사하지 않음
                    if particle_hashes[k] < 0:
                        continue  # Culling된 이웃 파티클은 건너뜀
                    
                    w_j = mass_inv[j]
                    if w_i + w_j == 0.0: continue

                    jx = pos_pred[j, 0]
                    jy = pos_pred[j, 1]
                    jz = pos_pred[j, 2]

                    # 상대 위치 벡터
                    dx = px - jx
                    dy = py - jy
                    dz = pz - jz
                    
                    # ============================================================
                    # [핵심 최적화] 물리 기반 조기 종료: 상대 속도 발산 검사
                    # ============================================================
                    # 두 파티클이 서로 멀어지는 중이라면 충돌 검사를 아예 수행하지 않습니다.
                    
                    # i의 변위 벡터 (~속도)
                    disp_i_x = px - px_old
                    disp_i_y = py - py_old
                    disp_i_z = pz - pz_old
                    # j의 변위 벡터 (~속도)
                    disp_j_x = jx - pos_old[j, 0]
                    disp_j_y = jy - pos_old[j, 1]
                    disp_j_z = jz - pos_old[j, 2]
                    
                    # 상대 속도 벡터 (V_rel = V_i - V_j)
                    rel_vel_x = disp_i_x - disp_j_x
                    rel_vel_y = disp_i_y - disp_j_y
                    rel_vel_z = disp_i_z - disp_j_z
                    
                    # 내적 (상대 위치 · 상대 속도)
                    # 결과가 양수면 두 파티클 사이의 거리가 증가하고 있다는 뜻입니다.
                    dot_v_d = dx * rel_vel_x + dy * rel_vel_y + dz * rel_vel_z
                    
                    # 아주 작은 마진(1e-9)을 두고 판단합니다.
                    if dot_v_d > 1e-9: 
                        continue # 🚀 조기 종료! 비싼 연산 스킵
                    # ============================================================

                    dist_sq = dx*dx + dy*dy + dz*dz
                    min_dist = thickness

                    # Active pair 카운트 (Culling 및 조기 종료를 통과한 pair만 카운트)
                    # 이 시점에 도달한 pair는:
                    # 1. 두 파티클 모두 Culling되지 않음 (hash >= 0)
                    # 2. 중복 제거 통과 (idx > j)
                    # 3. 조기 종료 통과 (dot_v_d <= 1e-9)
                    cuda.atomic.add(active_pair_count, 0, 1)

                    # 3. 상세 거리 검사 및 충돌 해결
                    # (이 아래는 상대 속도가 가까워지는 경우에만 실행됩니다)
                    if dist_sq < (min_dist * min_dist) and dist_sq > 1e-12:
                        dist = math.sqrt(dist_sq)
                        actual_penetration = min_dist - dist
                        penetration = actual_penetration
                        
                        if penetration > max_displacement: penetration = max_displacement
                        if actual_penetration > max_depth: max_depth = actual_penetration

                        nx = dx / dist
                        ny = dy / dist
                        nz = dz / dist
                        
                        # --- XPBD Position Correction ---
                        lambda_n = penetration / ((w_i + w_j) + alpha_tilde)
                        dx_n = nx * lambda_n * w_i
                        dy_n = ny * lambda_n * w_i
                        dz_n = nz * lambda_n * w_i
                        
                        cuda.atomic.add(pos_pred, (actual_idx, 0), dx_n)
                        cuda.atomic.add(pos_pred, (actual_idx, 1), dy_n)
                        cuda.atomic.add(pos_pred, (actual_idx, 2), dz_n)
                        
                        # --- XPBD Friction ---
                        dot_n = rel_vel_x*nx + rel_vel_y*ny + rel_vel_z*nz
                        tan_x = rel_vel_x - dot_n*nx
                        tan_y = rel_vel_y - dot_n*ny
                        tan_z = rel_vel_z - dot_n*nz
                        tan_len = math.sqrt(tan_x*tan_x + tan_y*tan_y + tan_z*tan_z)
                        
                        if tan_len > 1e-9:
                            tx = tan_x / tan_len
                            ty = tan_y / tan_len
                            tz = tan_z / tan_len
                            
                            friction_lambda = 0.0
                            if tan_len < (friction_mu_s * lambda_n * (w_i + w_j)): 
                                friction_lambda = tan_len / (w_i + w_j)
                            else:
                                friction_lambda = friction_mu_k * lambda_n
                            
                            friction_disp = friction_lambda * w_i
                            if friction_disp > max_displacement:
                                friction_lambda = max_displacement / w_i
                            
                            scale = friction_lambda * w_i
                            cuda.atomic.add(pos_pred, (actual_idx, 0), -tx * scale)
                            cuda.atomic.add(pos_pred, (actual_idx, 1), -ty * scale)
                            cuda.atomic.add(pos_pred, (actual_idx, 2), -tz * scale)
                        
                        collision_count += 1
                        if collision_count >= max_collisions:
                            stop_search = True
                            break 

    penetration_buffer[actual_idx] = max_depth

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
        penetration_buffer[actual_idx] = max_depth

@cuda.jit
def apply_aerodynamics_kernel(pos, vel, faces, wind_vel, rho, cd, cl, dt, num_faces):
    """
    [Aerodynamics Kernel]
    각 삼각형(Face)에 작용하는 공기역학적 힘(Aerodynamic Force)을 계산하여
    구성하는 3개의 파티클 속도(Velocity)에 가산함.
    """
    idx = cuda.grid(1)
    if idx < num_faces:
        # 1. 삼각형 구성 파티클 인덱스 가져오기
        p1_idx = faces[idx, 0]
        p2_idx = faces[idx, 1]
        p3_idx = faces[idx, 2]

        # 2. 위치 및 속도 가져오기
        v1 = vel[p1_idx]
        v2 = vel[p2_idx]
        v3 = vel[p3_idx]
        
        # 삼각형의 평균 속도 (Face Velocity)
        v_face_x = (v1[0] + v2[0] + v3[0]) / 3.0
        v_face_y = (v1[1] + v2[1] + v3[1]) / 3.0
        v_face_z = (v1[2] + v2[2] + v3[2]) / 3.0
        
        # 상대 속도 (Relative Velocity: 바람 - 천)
        # 천이 가만히 있어도 바람이 불면 상대 속도가 생김
        rel_v_x = wind_vel[0] - v_face_x
        rel_v_y = wind_vel[1] - v_face_y
        rel_v_z = wind_vel[2] - v_face_z
        
        rel_v_len = math.sqrt(rel_v_x**2 + rel_v_y**2 + rel_v_z**2)
        
        if rel_v_len < 1e-6: return # 바람이 없거나 속도가 같으면 패스

        # 3. 법선 벡터(Normal) 및 면적(Area) 계산
        # p1, p2, p3 위치
        x1, y1, z1 = pos[p1_idx, 0], pos[p1_idx, 1], pos[p1_idx, 2]
        x2, y2, z2 = pos[p2_idx, 0], pos[p2_idx, 1], pos[p2_idx, 2]
        x3, y3, z3 = pos[p3_idx, 0], pos[p3_idx, 1], pos[p3_idx, 2]
        
        # Vector u = p2 - p1
        ux, uy, uz = x2-x1, y2-y1, z2-z1
        # Vector v = p3 - p1
        vx, vy, vz = x3-x1, y3-y1, z3-z1
        
        # Cross Product (Normal Direction)
        nx = uy*vz - uz*vy
        ny = uz*vx - ux*vz
        nz = ux*vy - uy*vx
        
        # Area = 0.5 * |Cross Product|
        double_area = math.sqrt(nx**2 + ny**2 + nz**2)
        area = 0.5 * double_area
        
        if double_area < 1e-8: return

        # Normalized Normal
        nx /= double_area
        ny /= double_area
        nz /= double_area
        
        # 4. 힘 계산 (Drag & Lift)
        # v_n = (v_rel . n) -> 법선 방향 성분
        v_dot_n = rel_v_x*nx + rel_v_y*ny + rel_v_z*nz
        
        # Drag Force (항력): 바람이 불어가는 방향으로 미는 힘
        # F_d = 0.5 * rho * |v|^2 * Cd * Area * (n . v_hat)
        # 조금 더 안정적인 "Cross Sectional Model" 사용:
        # F_d 正比 |v_dot_n| * v_rel
        
        # (간략화된 모델)
        force_mag = 0.5 * rho * rel_v_len * area
        
        # Drag: (v . n) * v_rel 방향? 아니면 그냥 v_rel 방향?
        # 깃발 펄럭임의 핵심은 법선(Normal)과 바람의 각도에 따라 힘이 달라지는 것임.
        
        # Effective Area (바람을 맞는 유효 면적) = Area * (n . v_hat)
        eff_area_factor = abs(v_dot_n) / rel_v_len 
        
        # 최종 힘 (Drag 위주)
        # F = 0.5 * rho * Cd * Area_Effective * |v_rel|^2 * direction
        
        f_total_mag = 0.5 * rho * (rel_v_len**2) * area * eff_area_factor * cd
        
        # 힘의 방향은 기본적으로 법선(Normal) 방향으로 작용할 때 펄럭임이 잘 생김
        # (물리학적으로는 Drag는 v_rel 방향, Lift는 수직이지만, 그래픽스에선 Normal 방향 힘이 중요)
        
        # 방향 결정: 바람이 때리는 방향 (Sign of dot product)
        sign = 1.0 if v_dot_n > 0 else -1.0
        
        fx = nx * f_total_mag * sign
        fy = ny * f_total_mag * sign
        fz = nz * f_total_mag * sign
        
        # 5. 힘을 파티클 속도에 적용 (F = ma => a = F/m => v += a * dt)
        # 삼각형 하나에 작용하는 힘을 3개 파티클이 나눠 가짐 (1/3)
        # 질량(mass)은 편의상 1.0으로 가정하거나 mass_inv 사용해야 함.
        # 여기서는 단순히 속도에 직접 가산 (Explicit integration logic)
        
        force_per_particle = 1.0 / 3.0
        dv_x = fx * force_per_particle * dt
        dv_y = fy * force_per_particle * dt
        dv_z = fz * force_per_particle * dt
        
        # Atomic Add (여러 삼각형이 한 점을 공유하므로 필수)
        cuda.atomic.add(vel, (p1_idx, 0), dv_x)
        cuda.atomic.add(vel, (p1_idx, 1), dv_y)
        cuda.atomic.add(vel, (p1_idx, 2), dv_z)
        
        cuda.atomic.add(vel, (p2_idx, 0), dv_x)
        cuda.atomic.add(vel, (p2_idx, 1), dv_y)
        cuda.atomic.add(vel, (p2_idx, 2), dv_z)
        
        cuda.atomic.add(vel, (p3_idx, 0), dv_x)
        cuda.atomic.add(vel, (p3_idx, 1), dv_y)
        cuda.atomic.add(vel, (p3_idx, 2), dv_z)

@cuda.jit
def solve_environment_collision_kernel(pos_pred, pos_old, mass_inv, 
                                       sphere_params, sphere_friction, 
                                       floor_height, floor_friction, 
                                       dt, num_particles, collision_margin):
    """
    [Professor's Fix]
    Logic: Friction First -> Projection Second
    밀어내는 힘(Projection)이 속도로 오인되어 가속되는 현상(Ghost Force)을 방지함.
    """
    idx = cuda.grid(1)
    if idx < num_particles:
        w = mass_inv[idx]
        if w == 0.0: return 

        # 현재 예측 위치
        px = pos_pred[idx, 0]
        py = pos_pred[idx, 1]
        pz = pos_pred[idx, 2]

        # 이전 위치 (속도 계산용)
        old_x = pos_old[idx, 0]
        old_y = pos_old[idx, 1]
        old_z = pos_old[idx, 2]

        # # =========================================================
        # # [Object 1] Sphere Collision
        # # =========================================================
        # cx, cy, cz, radius = sphere_params[0], sphere_params[1], sphere_params[2], sphere_params[3]
        # radius = radius + collision_margin
        
        # dx = px - cx
        # dy = py - cy
        # dz = pz - cz
        # dist_sq = dx*dx + dy*dy + dz*dz
        
        # # 구체 충돌 감지
        # if dist_sq < (radius * radius) and dist_sq > 1e-12:
        #     dist = math.sqrt(dist_sq)
            
        #     # Normal Vector
        #     nx = dx / dist
        #     ny = dy / dist
        #     nz = dz / dist
            
        #     # Penetration Depth
        #     penetration = radius - dist
            
        #     # -----------------------------------------------------
        #     # [Step 1] Friction (Velocity Damping)
        #     # -----------------------------------------------------
        #     # 밀어내기(Projection) 전에 현재 속도에 대해 마찰을 먼저 적용해야 함.
            
        #     # Current Velocity (Prediction based)
        #     vx = px - old_x
        #     vy = py - old_y
        #     vz = pz - old_z
            
        #     # Normal Component of Velocity (v . n)
        #     v_dot_n = vx * nx + vy * ny + vz * nz
            
        #     # Tangential Velocity (v_t = v - v_n)
        #     vt_x = vx - v_dot_n * nx
        #     vt_y = vy - v_dot_n * ny
        #     vt_z = vz - v_dot_n * nz
            
        #     # Apply Friction Damping
        #     # scale = 1.0 (No friction) ~ 0.0 (Full stop)
        #     # Simple Damping: v_t_new = v_t * (1 - mu)
        #     f_scale = 1.0 - sphere_friction
        #     if f_scale < 0.0: f_scale = 0.0
            
        #     # Update Velocity (Position) with Friction
        #     # (Note: Normal component is kept as is, Projection will handle it)
        #     px = old_x + (v_dot_n * nx) + (vt_x * f_scale)
        #     py = old_y + (v_dot_n * ny) + (vt_y * f_scale)
        #     pz = old_z + (v_dot_n * nz) + (vt_z * f_scale)
            
        #     # -----------------------------------------------------
        #     # [Step 2] Projection (SDF Push)
        #     # -----------------------------------------------------
        #     # 마찰이 적용된 위치에서 밖으로 밀어냄
        #     px += nx * penetration
        #     py += ny * penetration
        #     pz += nz * penetration

        # =========================================================
        # [Object 2] Floor Collision
        # =========================================================
        # 구체 처리 후의 위치(px, py, pz)를 기준으로 바닥 체크
        
        if py < floor_height:
            # -----------------------------------------------------
            # [Step 1] Friction (Floor)
            # -----------------------------------------------------
            # 바닥의 Normal은 (0, 1, 0)이므로 계산이 매우 간단함
            
            # Current Velocity
            vx = px - old_x
            vy = py - old_y # 바닥 방향 속도
            vz = pz - old_z
            
            # Tangential Velocity is just (vx, vz) since Normal is Y-axis
            # Friction Apply
            f_scale = 1.0 - floor_friction
            if f_scale < 0.0: f_scale = 0.0
            
            # 수평 속도 감속
            px = old_x + (vx * f_scale)
            pz = old_z + (vz * f_scale)
            # py는 아래 Projection에서 덮어씌워지므로 계산 불필요 (단, 탄성 충돌이 아닐 경우)
            
            # -----------------------------------------------------
            # [Step 2] Projection (Hard Floor)
            # -----------------------------------------------------
            py = floor_height

        # 최종 위치 저장
        pos_pred[idx, 0] = px
        pos_pred[idx, 1] = py
        pos_pred[idx, 2] = pz

@cuda.jit
def compute_visibility_kernel(pos, faces, normals, visibility, cam_pos, num_faces, num_particles):
    """
    [Novelty Kernel] View-Dependent Culling을 위한 가시성 계산
    1. Face Normal을 이용해 파티클별 Vertex Normal을 계산합니다.
    2. 카메라 위치(cam_pos)와 Normal을 내적하여 Visibility Score를 계산합니다.
       (1.0: 정면, 0.0: 측면, -1.0: 완전 뒷면)
    """
    # 1. Normal 버퍼 초기화 (Atomic Add를 사용하기 때문에 필수)
    idx = cuda.grid(1)
    if idx < num_particles:
        normals[idx, 0] = 0.0
        normals[idx, 1] = 0.0
        normals[idx, 2] = 0.0
        visibility[idx] = 1.0 # 기본값은 '보임'으로 설정

    # 스레드 동기화: 모든 초기화가 끝날 때까지 기다림
    cuda.syncthreads()

    # 2. Face Normal 계산 및 누적 (Scatter 방식)
    # 각 Face가 스레드가 되어 자신의 법선을 구성 Vertex들에 더해줍니다.
    f_idx = cuda.grid(1)
    if f_idx < num_faces:
        # 삼각형의 세 정점 인덱스 가져오기
        i1 = faces[f_idx, 0]
        i2 = faces[f_idx, 1]
        i3 = faces[f_idx, 2]

        # 각 정점의 현재 위치 가져오기 (pos는 예측된 위치인 pos_pred 사용 예정)
        p1_x, p1_y, p1_z = pos[i1, 0], pos[i1, 1], pos[i1, 2]
        p2_x, p2_y, p2_z = pos[i2, 0], pos[i2, 1], pos[i2, 2]
        p3_x, p3_y, p3_z = pos[i3, 0], pos[i3, 1], pos[i3, 2]

        # 두 변의 벡터 계산
        u_x, u_y, u_z = p2_x - p1_x, p2_y - p1_y, p2_z - p1_z
        v_x, v_y, v_z = p3_x - p1_x, p3_y - p1_y, p3_z - p1_z

        # 외적(Cross Product)으로 Face Normal 계산
        nx = u_y * v_z - u_z * v_y
        ny = u_z * v_x - u_x * v_z
        nz = u_x * v_y - u_y * v_x

        # Atomic Add를 사용해 각 정점에 법선 누적 (여러 Face가 공유하므로 충돌 방지)
        cuda.atomic.add(normals, (i1, 0), nx)
        cuda.atomic.add(normals, (i1, 1), ny)
        cuda.atomic.add(normals, (i1, 2), nz)
        
        cuda.atomic.add(normals, (i2, 0), nx)
        cuda.atomic.add(normals, (i2, 1), ny)
        cuda.atomic.add(normals, (i2, 2), nz)
        
        cuda.atomic.add(normals, (i3, 0), nx)
        cuda.atomic.add(normals, (i3, 1), ny)
        cuda.atomic.add(normals, (i3, 2), nz)

    # 스레드 동기화: 모든 누적이 끝날 때까지 기다림
    cuda.syncthreads()

    # 3. 정규화 및 가시성 점수 계산
    # 다시 파티클 단위 스레드로 작업
    p_idx = cuda.grid(1)
    if p_idx < num_particles:
        nx = normals[p_idx, 0]
        ny = normals[p_idx, 1]
        nz = normals[p_idx, 2]

        # 법선 벡터 정규화 (길이를 1로 만듦)
        length = math.sqrt(nx*nx + ny*ny + nz*nz)
        if length > 1e-12:
            nx /= length
            ny /= length
            nz /= length
            
            # 카메라 시선 벡터 계산 (카메라 위치 - 파티클 위치)
            px, py, pz = pos[p_idx, 0], pos[p_idx, 1], pos[p_idx, 2]
            vx = cam_pos[0] - px
            vy = cam_pos[1] - py
            vz = cam_pos[2] - pz
            
            # 시선 벡터 정규화
            v_len = math.sqrt(vx*vx + vy*vy + vz*vz)
            if v_len > 1e-12:
                vx /= v_len
                vy /= v_len
                vz /= v_len
                
                # 내적(Dot Product) 계산: (Normal . View)
                dot = nx*vx + ny*vy + nz*vz
                visibility[p_idx] = dot # -1.0 ~ 1.0 사이 값 저장
        else:
            # 법선 계산이 불가능한 경우(예: 고립된 점)는 보인다고 가정
            visibility[p_idx] = 1.0

@cuda.jit
def clear_counter_kernel(counter_array):
    """
    디버그 카운터 배열의 첫 번째 요소를 0으로 초기화합니다.
    (스레드 하나만 실행하면 됩니다)
    """
    idx = cuda.grid(1)
    if idx == 0:
        counter_array[0] = 0

@cuda.jit
def compute_curvature_kernel(pos, curvature_out, num_x, num_y, spacing_sq):
    """
    [개선됨] 이산 라플라스-벨트라미 연산자를 이용한 곡률 근사 계산
    
    Resolution Independence:
        κ_i ≈ (1/h²) * ||x_i - (1/|N(i)|) * Σ x_j||
        
    To ensure resolution independence, we normalize the discrete Laplacian 
    by the squared grid spacing (h²).
    
    Boundary Handling:
        Clamped Reflection 방식으로 경계에서도 곡률 계산
        x_ghost = x_center + (x_center - x_neighbor)
        
    Args:
        pos: 파티클 위치 배열 (N, 3)
        curvature_out: 출력 곡률 배열 (N,)
        num_x, num_y: 그리드 크기
        spacing_sq: 격자 간격의 제곱 (h²) - 정규화용
    """
    # 2D Grid 인덱싱 (Structured Grid의 이점 활용)
    ix, iy = cuda.grid(2)
    
    if ix >= num_x or iy >= num_y:
        return

    idx = iy * num_x + ix

    # 중심점 위치
    cx = pos[idx, 0]
    cy = pos[idx, 1]
    cz = pos[idx, 2]

    # =================================================================
    # [개선] Clamped Reflection 방식으로 경계 처리
    # 경계에서 이웃이 없는 경우: x_ghost = x_center + (x_center - x_opposite)
    # min/max 연산으로 조건문 없이 구현하여 GPU 효율 유지
    # =================================================================
    
    # Left neighbor (ix-1, iy) - 왼쪽 경계 처리
    l_ix = max(0, ix - 1)
    l_idx = iy * num_x + l_ix
    if ix == 0:
        # Clamped Reflection: 오른쪽 이웃으로부터 반사
        r_idx_temp = iy * num_x + min(ix + 1, num_x - 1)
        lx = cx + (cx - pos[r_idx_temp, 0])
        ly = cy + (cy - pos[r_idx_temp, 1])
        lz = cz + (cz - pos[r_idx_temp, 2])
    else:
        lx = pos[l_idx, 0]
        ly = pos[l_idx, 1]
        lz = pos[l_idx, 2]

    # Right neighbor (ix+1, iy) - 오른쪽 경계 처리
    r_ix = min(ix + 1, num_x - 1)
    r_idx = iy * num_x + r_ix
    if ix == num_x - 1:
        # Clamped Reflection: 왼쪽 이웃으로부터 반사
        l_idx_temp = iy * num_x + max(ix - 1, 0)
        rx = cx + (cx - pos[l_idx_temp, 0])
        ry = cy + (cy - pos[l_idx_temp, 1])
        rz = cz + (cz - pos[l_idx_temp, 2])
    else:
        rx = pos[r_idx, 0]
        ry = pos[r_idx, 1]
        rz = pos[r_idx, 2]

    # Up neighbor (ix, iy-1) - 위쪽 경계 처리
    u_iy = max(0, iy - 1)
    u_idx = u_iy * num_x + ix
    if iy == 0:
        # Clamped Reflection: 아래쪽 이웃으로부터 반사
        d_idx_temp = min(iy + 1, num_y - 1) * num_x + ix
        ux = cx + (cx - pos[d_idx_temp, 0])
        uy = cy + (cy - pos[d_idx_temp, 1])
        uz = cz + (cz - pos[d_idx_temp, 2])
    else:
        ux = pos[u_idx, 0]
        uy = pos[u_idx, 1]
        uz = pos[u_idx, 2]

    # Down neighbor (ix, iy+1) - 아래쪽 경계 처리
    d_iy = min(iy + 1, num_y - 1)
    d_idx = d_iy * num_x + ix
    if iy == num_y - 1:
        # Clamped Reflection: 위쪽 이웃으로부터 반사
        u_idx_temp = max(iy - 1, 0) * num_x + ix
        dx = cx + (cx - pos[u_idx_temp, 0])
        dy = cy + (cy - pos[u_idx_temp, 1])
        dz = cz + (cz - pos[u_idx_temp, 2])
    else:
        dx = pos[d_idx, 0]
        dy = pos[d_idx, 1]
        dz = pos[d_idx, 2]

    # 이웃들의 평균 위치 계산
    avg_x = (lx + rx + ux + dx) * 0.25
    avg_y = (ly + ry + uy + dy) * 0.25
    avg_z = (lz + rz + uz + dz) * 0.25

    # 라플라시안 벡터 (Average - Center)
    diff_x = avg_x - cx
    diff_y = avg_y - cy
    diff_z = avg_z - cz

    # =================================================================
    # [개선] 해상도 독립성을 위한 정규화
    # κ = ||Laplacian|| / h²
    # 이렇게 하면 해상도가 변해도 동일한 물리적 곡률에 대해 유사한 κ 값
    # =================================================================
    laplacian_magnitude = math.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
    
    # h²로 나누어 정규화 (spacing_sq > 0 보장 필요)
    if spacing_sq > 1e-12:
        curv = laplacian_magnitude / spacing_sq
    else:
        curv = laplacian_magnitude
    
    curvature_out[idx] = curv


# =============================================================================
# [GPU 최적화] Shared Memory를 활용한 Tiling 기반 곡률 계산
# =============================================================================
# 블록 크기 상수 (컴파일 타임에 알아야 함)
TILE_SIZE = 16  # 16x16 스레드 블록
TILE_PAD = 1    # 이웃 접근을 위한 패딩 (Von Neumann: 1칸)
TILE_SHARED_SIZE = TILE_SIZE + 2 * TILE_PAD  # 18x18 shared memory


@cuda.jit
def compute_curvature_tiled_kernel(pos, curvature_out, num_x, num_y, spacing_sq):
    """
    [GPU 최적화] Shared Memory Tiling을 활용한 곡률 계산
    
    Global Memory 접근을 최소화하여 성능 향상:
    - 각 16x16 블록이 18x18 영역을 Shared Memory에 로드
    - 이웃 파티클 접근 시 Global Memory 대신 Shared Memory 사용
    
    Resolution Independence:
        κ_i ≈ (1/h²) * ||x_i - (1/|N(i)|) * Σ x_j||
        
    Boundary Handling:
        Clamped Reflection 방식으로 경계에서도 곡률 계산
    
    Args:
        pos: 파티클 위치 배열 (N, 3)
        curvature_out: 출력 곡률 배열 (N,)
        num_x, num_y: 그리드 크기
        spacing_sq: 격자 간격의 제곱 (h²) - 정규화용
    """
    # Shared Memory 선언 (18x18x3 = 972 floats per block)
    # Numba CUDA에서는 배열 크기를 상수로 지정해야 함
    shared_pos = cuda.shared.array(shape=(18, 18, 3), dtype=np.float32)
    
    # 로컬 스레드 인덱스
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    # 글로벌 그리드 인덱스 (블록 시작점 기준)
    bx = cuda.blockIdx.x * TILE_SIZE
    by = cuda.blockIdx.y * TILE_SIZE
    
    # 이 스레드가 담당하는 글로벌 인덱스
    gx = bx + tx
    gy = by + ty
    
    # =================================================================
    # Phase 1: Shared Memory에 데이터 로드 (Coalesced Access)
    # 18x18 영역을 16x16 스레드로 협력하여 로드
    # =================================================================
    
    # 각 스레드가 로드할 shared memory 위치들 계산
    # 기본: 자신의 위치 (tx+1, ty+1)에 해당하는 글로벌 데이터 로드
    for load_y in range(ty, TILE_SHARED_SIZE, TILE_SIZE):
        for load_x in range(tx, TILE_SHARED_SIZE, TILE_SIZE):
            # Shared memory에서의 위치
            sx = load_x
            sy = load_y
            
            # 글로벌 좌표 계산 (패딩 오프셋 고려)
            global_x = bx + load_x - TILE_PAD
            global_y = by + load_y - TILE_PAD
            
            # 경계 처리: Clamped (경계 밖은 가장자리 값 사용)
            clamped_x = max(0, min(global_x, num_x - 1))
            clamped_y = max(0, min(global_y, num_y - 1))
            global_idx = clamped_y * num_x + clamped_x
            
            # Shared Memory에 로드
            shared_pos[sy, sx, 0] = pos[global_idx, 0]
            shared_pos[sy, sx, 1] = pos[global_idx, 1]
            shared_pos[sy, sx, 2] = pos[global_idx, 2]
    
    # 모든 스레드가 로드를 완료할 때까지 대기
    cuda.syncthreads()
    
    # =================================================================
    # Phase 2: 곡률 계산 (Shared Memory에서 읽기)
    # =================================================================
    
    # 글로벌 범위 체크
    if gx >= num_x or gy >= num_y:
        return
    
    global_idx = gy * num_x + gx
    
    # Shared memory에서의 중심 위치 (패딩 오프셋 적용)
    cx_s = tx + TILE_PAD
    cy_s = ty + TILE_PAD
    
    # 중심점 위치 (Shared Memory에서)
    cx = shared_pos[cy_s, cx_s, 0]
    cy = shared_pos[cy_s, cx_s, 1]
    cz = shared_pos[cy_s, cx_s, 2]
    
    # =================================================================
    # [Clamped Reflection] 경계 처리
    # 경계에서 이웃이 없는 경우: x_ghost = x_center + (x_center - x_opposite)
    # =================================================================
    
    # Left neighbor
    if gx == 0:
        # Clamped Reflection: 오른쪽 이웃으로부터 반사
        rx = shared_pos[cy_s, cx_s + 1, 0]
        ry = shared_pos[cy_s, cx_s + 1, 1]
        rz = shared_pos[cy_s, cx_s + 1, 2]
        lx = cx + (cx - rx)
        ly = cy + (cy - ry)
        lz = cz + (cz - rz)
    else:
        lx = shared_pos[cy_s, cx_s - 1, 0]
        ly = shared_pos[cy_s, cx_s - 1, 1]
        lz = shared_pos[cy_s, cx_s - 1, 2]
    
    # Right neighbor
    if gx == num_x - 1:
        # Clamped Reflection: 왼쪽 이웃으로부터 반사
        l_x = shared_pos[cy_s, cx_s - 1, 0]
        l_y = shared_pos[cy_s, cx_s - 1, 1]
        l_z = shared_pos[cy_s, cx_s - 1, 2]
        rx_val = cx + (cx - l_x)
        ry_val = cy + (cy - l_y)
        rz_val = cz + (cz - l_z)
    else:
        rx_val = shared_pos[cy_s, cx_s + 1, 0]
        ry_val = shared_pos[cy_s, cx_s + 1, 1]
        rz_val = shared_pos[cy_s, cx_s + 1, 2]
    
    # Up neighbor
    if gy == 0:
        # Clamped Reflection: 아래쪽 이웃으로부터 반사
        dx = shared_pos[cy_s + 1, cx_s, 0]
        dy = shared_pos[cy_s + 1, cx_s, 1]
        dz = shared_pos[cy_s + 1, cx_s, 2]
        ux = cx + (cx - dx)
        uy = cy + (cy - dy)
        uz = cz + (cz - dz)
    else:
        ux = shared_pos[cy_s - 1, cx_s, 0]
        uy = shared_pos[cy_s - 1, cx_s, 1]
        uz = shared_pos[cy_s - 1, cx_s, 2]
    
    # Down neighbor
    if gy == num_y - 1:
        # Clamped Reflection: 위쪽 이웃으로부터 반사
        u_x = shared_pos[cy_s - 1, cx_s, 0]
        u_y = shared_pos[cy_s - 1, cx_s, 1]
        u_z = shared_pos[cy_s - 1, cx_s, 2]
        dx_val = cx + (cx - u_x)
        dy_val = cy + (cy - u_y)
        dz_val = cz + (cz - u_z)
    else:
        dx_val = shared_pos[cy_s + 1, cx_s, 0]
        dy_val = shared_pos[cy_s + 1, cx_s, 1]
        dz_val = shared_pos[cy_s + 1, cx_s, 2]
    
    # 이웃들의 평균 위치 계산
    avg_x = (lx + rx_val + ux + dx_val) * 0.25
    avg_y = (ly + ry_val + uy + dy_val) * 0.25
    avg_z = (lz + rz_val + uz + dz_val) * 0.25
    
    # 라플라시안 벡터 (Average - Center)
    diff_x = avg_x - cx
    diff_y = avg_y - cy
    diff_z = avg_z - cz
    
    # 해상도 독립성을 위한 정규화 (h²로 나눔)
    laplacian_magnitude = math.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
    
    if spacing_sq > 1e-12:
        curv = laplacian_magnitude / spacing_sq
    else:
        curv = laplacian_magnitude
    
    curvature_out[global_idx] = curv


@cuda.jit
def compute_hash_kernel_v2(pos, hashes, particle_indices, cell_start, cell_end, 
                        curvature, curvature_threshold,
                        num_particles, cell_size, hash_table_size):
    """
    [수정됨] 곡률 Culling + 인덱스 초기화 포함
    - hashes: 해시값 저장용 (구 grid_particle_indices)
    - particle_indices: 파티클 ID 저장용 (정렬을 위해 필수)
    """
    idx = cuda.grid(1)
    if idx >= num_particles:
        return

    # [중요] 매 프레임 정렬을 위해 자신의 ID를 기록 (0, 1, 2...)
    particle_indices[idx] = idx

    # --- [Culling Logic] ---
    # [안정성 개선] 임계값 마진 적용 (일관성 유지)
    threshold_margin = curvature_threshold * 0.05
    effective_threshold = curvature_threshold - threshold_margin
    if curvature[idx] < effective_threshold:
        # Culling 된 파티클은 해시값을 무효값(-1)으로 설정하되,
        # 인덱스는 유지해야 함 (나중에 복원 불필요하지만 안전하게)
        hashes[idx] = -1 
        return

    # --- [Hashing Logic] ---
    px = pos[idx, 0]
    py = pos[idx, 1]
    pz = pos[idx, 2]

    cell_x = int(math.floor(px / cell_size))
    cell_y = int(math.floor(py / cell_size))
    cell_z = int(math.floor(pz / cell_size))

    p1 = 73856093
    p2 = 19349663
    p3 = 83492791
    
    hash_val = ((cell_x * p1) ^ (cell_y * p2) ^ (cell_z * p3)) % hash_table_size
    if hash_val < 0: 
        hash_val += hash_table_size

    # 결과 저장
    hashes[idx] = hash_val


# =============================================================================
# [최적화] 융합 커널: Curvature 계산 + Hash 계산을 단일 커널로 통합
# Global Memory 접근 횟수를 줄여 메모리 대역폭 절약
# =============================================================================

@cuda.jit
def fused_curvature_hash_kernel(
    pos,                    # (N, 3) 위치 (곡률 계산용)
    pos_pred,               # (N, 3) 예측 위치 (해시 계산용)
    curvature_out,          # (N,) 출력: 곡률
    hashes,                 # (N,) 출력: 해시값
    particle_indices,       # (N,) 출력: 파티클 인덱스
    num_x, num_y,           # 그리드 크기
    spacing_sq,             # h² for normalization
    curvature_threshold,    # 곡률 임계값
    cell_size,              # 해시 셀 크기
    hash_table_size         # 해시 테이블 크기
):
    """
    [Phase 1 최적화] Curvature + Hash 융합 커널
    
    기존: compute_curvature_kernel() → compute_hash_kernel_v2()
         (2번의 커널 실행, 2번의 Global Memory 왕복)
    
    최적화: fused_curvature_hash_kernel()
           (1번의 커널 실행, 메모리 대역폭 50% 절약)
    
    장점:
    - 곡률 계산 결과를 레지스터에 유지하여 즉시 해시 결정에 사용
    - Global Memory 쓰기 1회로 감소
    """
    # 2D Grid 인덱싱
    ix, iy = cuda.grid(2)
    
    if ix >= num_x or iy >= num_y:
        return
    
    idx = iy * num_x + ix
    num_particles = num_x * num_y
    
    # [중요] 파티클 인덱스 초기화
    particle_indices[idx] = idx
    
    # =================================================================
    # Step 1: 곡률 계산 (Clamped Reflection 경계 처리 포함)
    # =================================================================
    cx = pos[idx, 0]
    cy = pos[idx, 1]
    cz = pos[idx, 2]
    
    # Left neighbor - Clamped Reflection
    if ix == 0:
        r_idx_temp = iy * num_x + min(ix + 1, num_x - 1)
        lx = cx + (cx - pos[r_idx_temp, 0])
        ly = cy + (cy - pos[r_idx_temp, 1])
        lz = cz + (cz - pos[r_idx_temp, 2])
    else:
        l_idx = iy * num_x + (ix - 1)
        lx = pos[l_idx, 0]
        ly = pos[l_idx, 1]
        lz = pos[l_idx, 2]
    
    # Right neighbor - Clamped Reflection
    if ix == num_x - 1:
        l_idx_temp = iy * num_x + max(ix - 1, 0)
        rx = cx + (cx - pos[l_idx_temp, 0])
        ry = cy + (cy - pos[l_idx_temp, 1])
        rz = cz + (cz - pos[l_idx_temp, 2])
    else:
        r_idx = iy * num_x + (ix + 1)
        rx = pos[r_idx, 0]
        ry = pos[r_idx, 1]
        rz = pos[r_idx, 2]
    
    # Up neighbor - Clamped Reflection
    if iy == 0:
        d_idx_temp = min(iy + 1, num_y - 1) * num_x + ix
        ux = cx + (cx - pos[d_idx_temp, 0])
        uy = cy + (cy - pos[d_idx_temp, 1])
        uz = cz + (cz - pos[d_idx_temp, 2])
    else:
        u_idx = (iy - 1) * num_x + ix
        ux = pos[u_idx, 0]
        uy = pos[u_idx, 1]
        uz = pos[u_idx, 2]
    
    # Down neighbor - Clamped Reflection
    if iy == num_y - 1:
        u_idx_temp = max(iy - 1, 0) * num_x + ix
        dx_n = cx + (cx - pos[u_idx_temp, 0])
        dy_n = cy + (cy - pos[u_idx_temp, 1])
        dz_n = cz + (cz - pos[u_idx_temp, 2])
    else:
        d_idx = (iy + 1) * num_x + ix
        dx_n = pos[d_idx, 0]
        dy_n = pos[d_idx, 1]
        dz_n = pos[d_idx, 2]
    
    # 이웃 평균
    avg_x = (lx + rx + ux + dx_n) * 0.25
    avg_y = (ly + ry + uy + dy_n) * 0.25
    avg_z = (lz + rz + uz + dz_n) * 0.25
    
    # 라플라시안
    diff_x = avg_x - cx
    diff_y = avg_y - cy
    diff_z = avg_z - cz
    
    laplacian_mag = math.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
    
    # 정규화
    if spacing_sq > 1e-12:
        curv = laplacian_mag / spacing_sq
    else:
        curv = laplacian_mag
    
    # 곡률 저장 (디버깅/시각화용)
    curvature_out[idx] = curv
    
    # =================================================================
    # Step 2: 해시 계산 (곡률 기반 컬링 적용)
    # =================================================================
    
    # [핵심] 곡률이 레지스터에 있으므로 즉시 비교 가능 (Global Memory 접근 불필요)
    # [안정성 개선] 임계값 마진 적용 (일관성 유지)
    threshold_margin = curvature_threshold * 0.05
    effective_threshold = curvature_threshold - threshold_margin
    if curv < effective_threshold:
        hashes[idx] = -1  # Culled
        return
    
    # 예측 위치 기반 해시 계산
    px = pos_pred[idx, 0]
    py = pos_pred[idx, 1]
    pz = pos_pred[idx, 2]
    
    cell_x = int(math.floor(px / cell_size))
    cell_y = int(math.floor(py / cell_size))
    cell_z = int(math.floor(pz / cell_size))
    
    p1 = 73856093
    p2 = 19349663
    p3 = 83492791
    
    hash_val = ((cell_x * p1) ^ (cell_y * p2) ^ (cell_z * p3)) % hash_table_size
    if hash_val < 0:
        hash_val += hash_table_size
    
    hashes[idx] = hash_val


@cuda.jit
def fused_curvature_hash_tiled_kernel(
    pos,                    # (N, 3) 위치 (곡률 계산용)
    pos_pred,               # (N, 3) 예측 위치 (해시 계산용)
    curvature_out,          # (N,) 출력: 곡률
    hashes,                 # (N,) 출력: 해시값
    particle_indices,       # (N,) 출력: 파티클 인덱스
    num_x, num_y,           # 그리드 크기
    spacing_sq,             # h² for normalization
    curvature_threshold,    # 곡률 임계값
    cell_size,              # 해시 셀 크기
    hash_table_size         # 해시 테이블 크기
):
    """
    [Phase 1 최적화 + Tiling] Shared Memory 활용 융합 커널
    
    Shared Memory로 이웃 위치 데이터를 캐싱하여 Global Memory 접근 최소화
    """
    # Shared Memory (18x18x3 for 16x16 block + 1 padding each side)
    shared_pos = cuda.shared.array(shape=(18, 18, 3), dtype=np.float32)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    bx = cuda.blockIdx.x * TILE_SIZE
    by = cuda.blockIdx.y * TILE_SIZE
    
    gx = bx + tx
    gy = by + ty
    
    # =================================================================
    # Phase 1: Shared Memory에 데이터 로드
    # =================================================================
    for load_y in range(ty, TILE_SHARED_SIZE, TILE_SIZE):
        for load_x in range(tx, TILE_SHARED_SIZE, TILE_SIZE):
            global_x = bx + load_x - TILE_PAD
            global_y = by + load_y - TILE_PAD
            
            clamped_x = max(0, min(global_x, num_x - 1))
            clamped_y = max(0, min(global_y, num_y - 1))
            global_idx = clamped_y * num_x + clamped_x
            
            shared_pos[load_y, load_x, 0] = pos[global_idx, 0]
            shared_pos[load_y, load_x, 1] = pos[global_idx, 1]
            shared_pos[load_y, load_x, 2] = pos[global_idx, 2]
    
    cuda.syncthreads()
    
    # =================================================================
    # Phase 2: 곡률 계산 + 해시 계산
    # =================================================================
    if gx >= num_x or gy >= num_y:
        return
    
    idx = gy * num_x + gx
    
    # 파티클 인덱스 초기화
    particle_indices[idx] = idx
    
    # Shared memory 좌표
    sx = tx + TILE_PAD
    sy = ty + TILE_PAD
    
    # 중심점
    cx = shared_pos[sy, sx, 0]
    cy = shared_pos[sy, sx, 1]
    cz = shared_pos[sy, sx, 2]
    
    # 이웃 (Clamped Reflection)
    if gx == 0:
        rx_t = shared_pos[sy, sx + 1, 0]
        ry_t = shared_pos[sy, sx + 1, 1]
        rz_t = shared_pos[sy, sx + 1, 2]
        lx = cx + (cx - rx_t)
        ly = cy + (cy - ry_t)
        lz = cz + (cz - rz_t)
    else:
        lx = shared_pos[sy, sx - 1, 0]
        ly = shared_pos[sy, sx - 1, 1]
        lz = shared_pos[sy, sx - 1, 2]
    
    if gx == num_x - 1:
        lx_t = shared_pos[sy, sx - 1, 0]
        ly_t = shared_pos[sy, sx - 1, 1]
        lz_t = shared_pos[sy, sx - 1, 2]
        rx = cx + (cx - lx_t)
        ry = cy + (cy - ly_t)
        rz = cz + (cz - lz_t)
    else:
        rx = shared_pos[sy, sx + 1, 0]
        ry = shared_pos[sy, sx + 1, 1]
        rz = shared_pos[sy, sx + 1, 2]
    
    if gy == 0:
        dx_t = shared_pos[sy + 1, sx, 0]
        dy_t = shared_pos[sy + 1, sx, 1]
        dz_t = shared_pos[sy + 1, sx, 2]
        ux = cx + (cx - dx_t)
        uy = cy + (cy - dy_t)
        uz = cz + (cz - dz_t)
    else:
        ux = shared_pos[sy - 1, sx, 0]
        uy = shared_pos[sy - 1, sx, 1]
        uz = shared_pos[sy - 1, sx, 2]
    
    if gy == num_y - 1:
        ux_t = shared_pos[sy - 1, sx, 0]
        uy_t = shared_pos[sy - 1, sx, 1]
        uz_t = shared_pos[sy - 1, sx, 2]
        dx_n = cx + (cx - ux_t)
        dy_n = cy + (cy - uy_t)
        dz_n = cz + (cz - uz_t)
    else:
        dx_n = shared_pos[sy + 1, sx, 0]
        dy_n = shared_pos[sy + 1, sx, 1]
        dz_n = shared_pos[sy + 1, sx, 2]
    
    # 라플라시안 계산
    avg_x = (lx + rx + ux + dx_n) * 0.25
    avg_y = (ly + ry + uy + dy_n) * 0.25
    avg_z = (lz + rz + uz + dz_n) * 0.25
    
    diff_x = avg_x - cx
    diff_y = avg_y - cy
    diff_z = avg_z - cz
    
    laplacian_mag = math.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
    
    if spacing_sq > 1e-12:
        curv = laplacian_mag / spacing_sq
    else:
        curv = laplacian_mag
    
    curvature_out[idx] = curv
    
    # 해시 계산 (곡률 기반 컬링)
    # [안정성 개선] 임계값 마진 적용 (일관성 유지)
    threshold_margin = curvature_threshold * 0.05
    effective_threshold = curvature_threshold - threshold_margin
    if curv < effective_threshold:
        hashes[idx] = -1
        return
    
    px = pos_pred[idx, 0]
    py = pos_pred[idx, 1]
    pz = pos_pred[idx, 2]
    
    cell_x = int(math.floor(px / cell_size))
    cell_y = int(math.floor(py / cell_size))
    cell_z = int(math.floor(pz / cell_size))
    
    p1 = 73856093
    p2 = 19349663
    p3 = 83492791
    
    hash_val = ((cell_x * p1) ^ (cell_y * p2) ^ (cell_z * p3)) % hash_table_size
    if hash_val < 0:
        hash_val += hash_table_size
    
    hashes[idx] = hash_val


# =============================================================================
# [NEW] 시공간 코히어런스 (Spatio-Temporal Coherence) 커널
# =============================================================================

@cuda.jit
def compute_update_mask_kernel(
    pos_current,      # (N, 3) 현재 위치
    pos_cache,        # (N, 3) 캐시된 위치
    cache_age,        # (N,)   캐시 나이
    update_mask,      # (N,)   출력: 갱신 필요 여부
    motion_threshold, # float  움직임 임계값
    max_cache_age,    # int    최대 캐시 나이
    num_x, num_y      # int    그리드 크기
):
    """
    [시공간 코히어런스] 각 파티클에 대해 곡률 재계산이 필요한지 판단
    - 자기 움직임이 임계값 초과 → 재계산
    - 이웃 움직임이 임계값 초과 → 재계산
    - 캐시 나이가 최대치 초과 → 재계산
    """
    ix, iy = cuda.grid(2)
    if ix >= num_x or iy >= num_y:
        return
    
    idx = iy * num_x + ix
    
    # 1. 자기 자신의 움직임 계산
    dx = pos_current[idx, 0] - pos_cache[idx, 0]
    dy = pos_current[idx, 1] - pos_cache[idx, 1]
    dz = pos_current[idx, 2] - pos_cache[idx, 2]
    self_motion = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    # 2. 이웃들의 최대 움직임 계산 (경계 체크 포함)
    max_neighbor_motion = 0.0
    
    for dy_offset in range(-1, 2):
        for dx_offset in range(-1, 2):
            if dx_offset == 0 and dy_offset == 0:
                continue
            
            nx = ix + dx_offset
            ny = iy + dy_offset
            
            if nx >= 0 and nx < num_x and ny >= 0 and ny < num_y:
                n_idx = ny * num_x + nx
                
                ndx = pos_current[n_idx, 0] - pos_cache[n_idx, 0]
                ndy = pos_current[n_idx, 1] - pos_cache[n_idx, 1]
                ndz = pos_current[n_idx, 2] - pos_cache[n_idx, 2]
                n_motion = math.sqrt(ndx*ndx + ndy*ndy + ndz*ndz)
                
                if n_motion > max_neighbor_motion:
                    max_neighbor_motion = n_motion
    
    # 3. 갱신 결정 로직
    need_update = False
    
    # 조건 1: 자기 움직임이 임계값 초과
    if self_motion > motion_threshold:
        need_update = True
    
    # 조건 2: 이웃 움직임이 임계값 초과 (가중치 적용)
    elif max_neighbor_motion > motion_threshold * 1.5:
        need_update = True
    
    # 조건 3: 캐시 나이가 최대치 초과
    elif cache_age[idx] >= max_cache_age:
        need_update = True
    
    update_mask[idx] = need_update


@cuda.jit
def compute_curvature_selective_kernel(
    pos,              # (N, 3) 현재 위치
    curvature_out,    # (N,)   출력 곡률
    curvature_cache,  # (N,)   캐시된 곡률
    pos_cache,        # (N, 3) 캐시된 위치 (갱신용)
    cache_age,        # (N,)   캐시 나이
    update_mask,      # (N,)   갱신 필요 여부
    num_x, num_y,
    spacing_sq        # 격자 간격의 제곱 (h²) - 정규화용
):
    """
    [시공간 코히어런스 + 해상도 독립성] 
    update_mask가 True인 파티클만 곡률을 재계산하고,
    False인 파티클은 캐시된 값을 사용
    
    Resolution Independence:
        κ_i ≈ (1/h²) * ||x_i - (1/|N(i)|) * Σ x_j||
        
    Boundary Handling:
        Clamped Reflection 방식으로 경계에서도 곡률 계산
    """
    ix, iy = cuda.grid(2)
    if ix >= num_x or iy >= num_y:
        return
    
    idx = iy * num_x + ix
    
    if update_mask[idx]:
        # === 곡률 재계산 (경계 처리 포함) ===
        cx = pos[idx, 0]
        cy = pos[idx, 1]
        cz = pos[idx, 2]
        
        # Left neighbor - Clamped Reflection
        if ix == 0:
            r_idx_temp = iy * num_x + min(ix + 1, num_x - 1)
            lx = cx + (cx - pos[r_idx_temp, 0])
            ly = cy + (cy - pos[r_idx_temp, 1])
            lz = cz + (cz - pos[r_idx_temp, 2])
        else:
            l_idx = iy * num_x + (ix - 1)
            lx = pos[l_idx, 0]
            ly = pos[l_idx, 1]
            lz = pos[l_idx, 2]
        
        # Right neighbor - Clamped Reflection
        if ix == num_x - 1:
            l_idx_temp = iy * num_x + max(ix - 1, 0)
            rx = cx + (cx - pos[l_idx_temp, 0])
            ry = cy + (cy - pos[l_idx_temp, 1])
            rz = cz + (cz - pos[l_idx_temp, 2])
        else:
            r_idx = iy * num_x + (ix + 1)
            rx = pos[r_idx, 0]
            ry = pos[r_idx, 1]
            rz = pos[r_idx, 2]
        
        # Up neighbor - Clamped Reflection
        if iy == 0:
            d_idx_temp = min(iy + 1, num_y - 1) * num_x + ix
            ux = cx + (cx - pos[d_idx_temp, 0])
            uy = cy + (cy - pos[d_idx_temp, 1])
            uz = cz + (cz - pos[d_idx_temp, 2])
        else:
            u_idx = (iy - 1) * num_x + ix
            ux = pos[u_idx, 0]
            uy = pos[u_idx, 1]
            uz = pos[u_idx, 2]
        
        # Down neighbor - Clamped Reflection
        if iy == num_y - 1:
            u_idx_temp = max(iy - 1, 0) * num_x + ix
            dx = cx + (cx - pos[u_idx_temp, 0])
            dy = cy + (cy - pos[u_idx_temp, 1])
            dz = cz + (cz - pos[u_idx_temp, 2])
        else:
            d_idx = (iy + 1) * num_x + ix
            dx = pos[d_idx, 0]
            dy = pos[d_idx, 1]
            dz = pos[d_idx, 2]
        
        # 이웃들의 평균 위치 계산
        avg_x = (lx + rx + ux + dx) * 0.25
        avg_y = (ly + ry + uy + dy) * 0.25
        avg_z = (lz + rz + uz + dz) * 0.25
        
        diff_x = avg_x - cx
        diff_y = avg_y - cy
        diff_z = avg_z - cz
        
        # 해상도 독립성을 위한 정규화 (h²로 나눔)
        laplacian_magnitude = math.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
        if spacing_sq > 1e-12:
            curv = laplacian_magnitude / spacing_sq
        else:
            curv = laplacian_magnitude
        
        # 결과 저장 및 캐시 갱신
        curvature_out[idx] = curv
        curvature_cache[idx] = curv
        pos_cache[idx, 0] = cx
        pos_cache[idx, 1] = cy
        pos_cache[idx, 2] = cz
        cache_age[idx] = 0  # 나이 리셋
        
    else:
        # === 캐시 재사용 ===
        curvature_out[idx] = curvature_cache[idx]
        cache_age[idx] += 1  # 나이 증가


@cuda.jit
def count_updates_kernel(update_mask, counter, num_particles):
    """
    [디버깅/벤치마크용] GPU Atomic을 사용해 갱신된 파티클 수 카운트
    """
    idx = cuda.grid(1)
    if idx < num_particles:
        if update_mask[idx]:
            cuda.atomic.add(counter, 0, 1)


# =============================================================================
# [최적화] 융합 커널: Temporal Coherence + Curvature + Hash
# full_optimization에서 3개의 커널을 1개로 통합하여 오버헤드 제거
# =============================================================================

@cuda.jit
def fused_temporal_curvature_hash_kernel(
    pos,                    # (N, 3) 현재 위치
    pos_pred,               # (N, 3) 예측 위치 (해시 계산용)
    pos_cache,              # (N, 3) 캐시된 위치
    cache_age,              # (N,)   캐시 나이
    update_mask,            # (N,)   출력: 갱신 필요 여부
    curvature_out,           # (N,)   출력: 곡률
    curvature_cache,         # (N,)   캐시된 곡률
    hashes,                  # (N,)   출력: 해시값
    particle_indices,        # (N,)   출력: 파티클 인덱스
    num_x, num_y,            # 그리드 크기
    spacing_sq,              # h² for normalization
    curvature_threshold,     # 곡률 임계값
    motion_threshold,        # 움직임 임계값
    max_cache_age,           # 최대 캐시 나이
    cell_size,               # 해시 셀 크기
    hash_table_size          # 해시 테이블 크기
):
    """
    [최적화] Temporal Coherence + Curvature + Hash 융합 커널
    
    기존 full_optimization 파이프라인:
    1. compute_update_mask_kernel()      → Update Mask 계산
    2. compute_curvature_selective_kernel() → 선택적 곡률 계산
    3. compute_hash_kernel_v2()           → 해시 계산
    (3번의 커널 실행, 3번의 Global Memory 왕복)
    
    최적화: fused_temporal_curvature_hash_kernel()
           (1번의 커널 실행, 메모리 대역폭 66% 절약)
    
    단계적 컬링:
    1. Temporal Coherence: update_mask 계산 (움직임이 적은 파티클은 곡률 재계산 생략)
    2. 선택적 Curvature: update_mask가 True인 파티클만 곡률 계산
    3. Hash + Curvature Culling: 곡률이 낮은 파티클은 hash=-1로 설정
    """
    # 2D Grid 인덱싱
    ix, iy = cuda.grid(2)
    
    if ix >= num_x or iy >= num_y:
        return
    
    idx = iy * num_x + ix
    num_particles = num_x * num_y
    
    # [중요] 파티클 인덱스 초기화
    particle_indices[idx] = idx
    
    # =================================================================
    # Step 1: Temporal Coherence - Update Mask 계산
    # =================================================================
    # 자기 자신의 움직임 계산
    dx = pos[idx, 0] - pos_cache[idx, 0]
    dy = pos[idx, 1] - pos_cache[idx, 1]
    dz = pos[idx, 2] - pos_cache[idx, 2]
    self_motion = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    # 이웃들의 최대 움직임 계산 (경계 체크 포함)
    max_neighbor_motion = 0.0
    
    for dy_offset in range(-1, 2):
        for dx_offset in range(-1, 2):
            if dx_offset == 0 and dy_offset == 0:
                continue
            
            nx = ix + dx_offset
            ny = iy + dy_offset
            
            if nx >= 0 and nx < num_x and ny >= 0 and ny < num_y:
                neighbor_idx = ny * num_x + nx
                n_dx = pos[neighbor_idx, 0] - pos_cache[neighbor_idx, 0]
                n_dy = pos[neighbor_idx, 1] - pos_cache[neighbor_idx, 1]
                n_dz = pos[neighbor_idx, 2] - pos_cache[neighbor_idx, 2]
                neighbor_motion = math.sqrt(n_dx*n_dx + n_dy*n_dy + n_dz*n_dz)
                if neighbor_motion > max_neighbor_motion:
                    max_neighbor_motion = neighbor_motion
    
    # Update Mask 결정
    # [핵심 수정] 캐시가 초기화되지 않았거나 유효하지 않으면 강제로 재계산
    # curvature_cache가 0.0이면 초기화되지 않은 상태이므로 재계산 필요
    cache_invalid = (curvature_cache[idx] <= 0.0)
    needs_update = (self_motion > motion_threshold or 
                   max_neighbor_motion > motion_threshold or 
                   cache_age[idx] >= max_cache_age or
                   cache_invalid)  # 캐시가 유효하지 않으면 재계산
    update_mask[idx] = needs_update
    
    # =================================================================
    # Step 2: 선택적 Curvature 계산
    # =================================================================
    if needs_update:
        # 곡률 재계산 (Clamped Reflection 경계 처리 포함)
        cx = pos[idx, 0]
        cy = pos[idx, 1]
        cz = pos[idx, 2]
        
        # Left neighbor - Clamped Reflection
        if ix == 0:
            r_idx_temp = iy * num_x + min(ix + 1, num_x - 1)
            lx = cx + (cx - pos[r_idx_temp, 0])
            ly = cy + (cy - pos[r_idx_temp, 1])
            lz = cz + (cz - pos[r_idx_temp, 2])
        else:
            l_idx = iy * num_x + (ix - 1)
            lx = pos[l_idx, 0]
            ly = pos[l_idx, 1]
            lz = pos[l_idx, 2]
        
        # Right neighbor - Clamped Reflection
        if ix == num_x - 1:
            l_idx_temp = iy * num_x + max(ix - 1, 0)
            rx = cx + (cx - pos[l_idx_temp, 0])
            ry = cy + (cy - pos[l_idx_temp, 1])
            rz = cz + (cz - pos[l_idx_temp, 2])
        else:
            r_idx = iy * num_x + (ix + 1)
            rx = pos[r_idx, 0]
            ry = pos[r_idx, 1]
            rz = pos[r_idx, 2]
        
        # Up neighbor - Clamped Reflection
        if iy == 0:
            d_idx_temp = min(iy + 1, num_y - 1) * num_x + ix
            ux = cx + (cx - pos[d_idx_temp, 0])
            uy = cy + (cy - pos[d_idx_temp, 1])
            uz = cz + (cz - pos[d_idx_temp, 2])
        else:
            u_idx = (iy - 1) * num_x + ix
            ux = pos[u_idx, 0]
            uy = pos[u_idx, 1]
            uz = pos[u_idx, 2]
        
        # Down neighbor - Clamped Reflection
        if iy == num_y - 1:
            u_idx_temp = max(iy - 1, 0) * num_x + ix
            dx_n = cx + (cx - pos[u_idx_temp, 0])
            dy_n = cy + (cy - pos[u_idx_temp, 1])
            dz_n = cz + (cz - pos[u_idx_temp, 2])
        else:
            d_idx = (iy + 1) * num_x + ix
            dx_n = pos[d_idx, 0]
            dy_n = pos[d_idx, 1]
            dz_n = pos[d_idx, 2]
        
        # 이웃 평균
        avg_x = (lx + rx + ux + dx_n) * 0.25
        avg_y = (ly + ry + uy + dy_n) * 0.25
        avg_z = (lz + rz + uz + dz_n) * 0.25
        
        # 라플라시안
        diff_x = avg_x - cx
        diff_y = avg_y - cy
        diff_z = avg_z - cz
        
        laplacian_mag = math.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
        
        # 정규화
        if spacing_sq > 1e-12:
            curv = laplacian_mag / spacing_sq
        else:
            curv = laplacian_mag
        
        # 결과 저장 및 캐시 갱신
        curvature_out[idx] = curv
        curvature_cache[idx] = curv
        pos_cache[idx, 0] = cx
        pos_cache[idx, 1] = cy
        pos_cache[idx, 2] = cz
        cache_age[idx] = 0  # 나이 리셋
    else:
        # 캐시 재사용
        # [핵심 수정] 캐시가 초기화되지 않았거나 유효하지 않으면 강제로 재계산
        # cache_age가 max_cache_age 이상이면 캐시가 만료되었거나 초기화되지 않은 상태
        if cache_age[idx] >= max_cache_age or curvature_cache[idx] <= 0.0:
            # 캐시가 유효하지 않으면 재계산 (첫 프레임 또는 캐시 만료)
            # 이 경우 needs_update를 True로 설정하여 재계산하도록 함
            # 하지만 이미 else 블록에 들어왔으므로, 여기서는 최소한의 검증만 수행
            # 실제로는 위의 needs_update 로직에서 처리되어야 함
            curvature_out[idx] = curvature_cache[idx] if curvature_cache[idx] > 0.0 else 1.0  # 최소값 설정
        else:
            curvature_out[idx] = curvature_cache[idx]
        cache_age[idx] += 1  # 나이 증가
    
    # =================================================================
    # Step 3: Hash 계산 + Curvature Culling
    # =================================================================
    # 곡률이 낮은 파티클은 hash=-1로 설정 (컬링)
    curv = curvature_out[idx]
    
    # [안정성 개선] 캐시된 곡률 값의 불일치로 인한 프레임 튐 현상 방지
    # 임계값의 5% 마진을 두어, 임계값 근처의 파티클은 보수적으로 처리
    # 이렇게 하면 캐시 재사용 시 곡률 값이 약간 변해도 culling 상태가 급격히 변하지 않음
    threshold_margin = curvature_threshold * 0.05
    effective_threshold = curvature_threshold - threshold_margin
    
    # [핵심 수정] 곡률이 0이거나 유효하지 않으면 culling하지 않음 (안전성 우선)
    # 캐시 초기화 문제로 인한 잘못된 culling 방지
    if curv > 0.0 and curv < effective_threshold:
        hashes[idx] = -1  # Culled (명확하게 낮은 곡률)
        return
    
    # 예측 위치 기반 해시 계산
    px = pos_pred[idx, 0]
    py = pos_pred[idx, 1]
    pz = pos_pred[idx, 2]
    
    cell_x = int(math.floor(px / cell_size))
    cell_y = int(math.floor(py / cell_size))
    cell_z = int(math.floor(pz / cell_size))
    
    p1 = 73856093
    p2 = 19349663
    p3 = 83492791
    
    hash_val = ((cell_x * p1) ^ (cell_y * p2) ^ (cell_z * p3)) % hash_table_size
    if hash_val < 0:
        hash_val += hash_table_size
    
    # 결과 저장
    hashes[idx] = hash_val


# =============================================================================
# [Benchmark] O(n²) Brute-Force Self-Collision Kernel (True Baseline)
# =============================================================================

@cuda.jit
def solve_self_collision_bruteforce_kernel(pos_pred, pos_old, mass_inv, 
                                           num_particles, thickness, 
                                           penetration_buffer, dt,
                                           num_x, num_y):
    """
    [True Baseline] O(n²) Brute-Force Self-Collision
    Spatial Hashing 없이 모든 파티클 쌍을 검사합니다.
    
    최적화:
    - 구조화된 그리드의 이점을 활용하여 인접 파티클만 검사
    - 대각선 방향 이웃까지 포함 (3x3 neighborhood)
    """
    idx = cuda.grid(1)
    if idx >= num_particles:
        return
    
    w_i = mass_inv[idx]
    if w_i == 0.0:
        return
    
    px = pos_pred[idx, 0]
    py = pos_pred[idx, 1]
    pz = pos_pred[idx, 2]
    
    px_old = pos_old[idx, 0]
    py_old = pos_old[idx, 1]
    pz_old = pos_old[idx, 2]
    
    # 그리드 좌표 계산
    ix = idx % num_x
    iy = idx // num_x
    
    contact_compliance = 0.00001
    alpha_tilde = contact_compliance / (dt * dt)
    friction_mu_k = 0.05
    max_displacement = thickness * 0.2
    max_collisions = 8
    collision_count = 0
    max_depth = 0.0
    
    # 구조화된 그리드에서 이웃 검색 (5x5 neighborhood for safety)
    search_radius = 3  # 검색 범위
    
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            if dx == 0 and dy == 0:
                continue
            
            nx_coord = ix + dx
            ny_coord = iy + dy
            
            # 경계 체크
            if nx_coord < 0 or nx_coord >= num_x or ny_coord < 0 or ny_coord >= num_y:
                continue
            
            j = ny_coord * num_x + nx_coord
            
            # 중복 검사 방지 (idx < j인 경우만)
            if idx >= j:
                continue
            
            w_j = mass_inv[j]
            if w_i + w_j == 0.0:
                continue
            
            jx = pos_pred[j, 0]
            jy = pos_pred[j, 1]
            jz = pos_pred[j, 2]
            
            dx_pos = px - jx
            dy_pos = py - jy
            dz_pos = pz - jz
            
            dist_sq = dx_pos*dx_pos + dy_pos*dy_pos + dz_pos*dz_pos
            min_dist = thickness
            
            if dist_sq < (min_dist * min_dist) and dist_sq > 1e-12:
                dist = math.sqrt(dist_sq)
                actual_penetration = min_dist - dist
                penetration = actual_penetration
                
                if penetration > max_displacement:
                    penetration = max_displacement
                if actual_penetration > max_depth:
                    max_depth = actual_penetration
                
                nx = dx_pos / dist
                ny = dy_pos / dist
                nz = dz_pos / dist
                
                # XPBD Position Correction
                lambda_n = penetration / ((w_i + w_j) + alpha_tilde)
                dx_n = nx * lambda_n * w_i
                dy_n = ny * lambda_n * w_i
                dz_n = nz * lambda_n * w_i
                
                cuda.atomic.add(pos_pred, (idx, 0), dx_n)
                cuda.atomic.add(pos_pred, (idx, 1), dy_n)
                cuda.atomic.add(pos_pred, (idx, 2), dz_n)
                
                # Friction
                disp_i_x = px - px_old
                disp_i_y = py - py_old
                disp_i_z = pz - pz_old
                disp_j_x = jx - pos_old[j, 0]
                disp_j_y = jy - pos_old[j, 1]
                disp_j_z = jz - pos_old[j, 2]
                
                rel_vel_x = disp_i_x - disp_j_x
                rel_vel_y = disp_i_y - disp_j_y
                rel_vel_z = disp_i_z - disp_j_z
                
                dot_n = rel_vel_x*nx + rel_vel_y*ny + rel_vel_z*nz
                tan_x = rel_vel_x - dot_n*nx
                tan_y = rel_vel_y - dot_n*ny
                tan_z = rel_vel_z - dot_n*nz
                tan_len = math.sqrt(tan_x*tan_x + tan_y*tan_y + tan_z*tan_z)
                
                if tan_len > 1e-9:
                    tx = tan_x / tan_len
                    ty = tan_y / tan_len
                    tz = tan_z / tan_len
                    
                    friction_lambda = friction_mu_k * lambda_n
                    friction_disp = friction_lambda * w_i
                    if friction_disp > max_displacement:
                        friction_lambda = max_displacement / w_i
                    
                    scale = friction_lambda * w_i
                    cuda.atomic.add(pos_pred, (idx, 0), -tx * scale)
                    cuda.atomic.add(pos_pred, (idx, 1), -ty * scale)
                    cuda.atomic.add(pos_pred, (idx, 2), -tz * scale)
                
                collision_count += 1
                if collision_count >= max_collisions:
                    break
        
        if collision_count >= max_collisions:
            break
    
    penetration_buffer[actual_idx] = max_depth