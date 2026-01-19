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
                                         visibility, frame_idx, debug_skip_count): 
    """
    [Sorted-based Spatial Hashing] + [Physics-based Early Termination] + [Conservative Culling]
    물리 기반 조기 종료 최적화를 적용하고, 컬링 파라미터를 안정적인 값으로 되돌립니다.
    """
    idx = cuda.grid(1)
    if idx >= num_particles: return

    # ============================================================
    # 1. [Novelty] View-Dependent Culling (안정성 복구)
    # ============================================================
    # 공격적인 파라미터로 인한 퀄리티 저하를 막기 위해 다시 보수적으로 설정합니다.
    vis_score = visibility[idx]
    culling_threshold = 0.2  # 0.5 -> 0.2 (뒷면 판단 기준 완화)

    if vis_score < culling_threshold:
        seed = idx * 12345 + frame_idx * 67897
        rand_state = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        rand_float = float(rand_state) / 2147483648.0 
        
        skip_probability = 0.7 # 0.95 -> 0.7 (스킵 확률 감소)

        if rand_float < skip_probability:
            # cuda.atomic.add(debug_skip_count, 0, 1)
            return
    # ============================================================
    
    w_i = mass_inv[idx]
    if w_i == 0.0: return
    
    px = pos_pred[idx, 0]
    py = pos_pred[idx, 1]
    pz = pos_pred[idx, 2]

    # 이전 프레임 위치 (속도 계산용)
    px_old = pos_old[idx, 0]
    py_old = pos_old[idx, 1]
    pz_old = pos_old[idx, 2]
    
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
                        
                        cuda.atomic.add(pos_pred, (idx, 0), dx_n)
                        cuda.atomic.add(pos_pred, (idx, 1), dy_n)
                        cuda.atomic.add(pos_pred, (idx, 2), dz_n)
                        
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
                            cuda.atomic.add(pos_pred, (idx, 0), -tx * scale)
                            cuda.atomic.add(pos_pred, (idx, 1), -ty * scale)
                            cuda.atomic.add(pos_pred, (idx, 2), -tz * scale)
                        
                        collision_count += 1
                        if collision_count >= max_collisions:
                            stop_search = True
                            break 

    penetration_buffer[idx] = max_depth

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

        # =========================================================
        # [Object 1] Sphere Collision
        # =========================================================
        cx, cy, cz, radius = sphere_params[0], sphere_params[1], sphere_params[2], sphere_params[3]
        radius = radius + collision_margin
        
        dx = px - cx
        dy = py - cy
        dz = pz - cz
        dist_sq = dx*dx + dy*dy + dz*dz
        
        # 구체 충돌 감지
        if dist_sq < (radius * radius) and dist_sq > 1e-12:
            dist = math.sqrt(dist_sq)
            
            # Normal Vector
            nx = dx / dist
            ny = dy / dist
            nz = dz / dist
            
            # Penetration Depth
            penetration = radius - dist
            
            # -----------------------------------------------------
            # [Step 1] Friction (Velocity Damping)
            # -----------------------------------------------------
            # 밀어내기(Projection) 전에 현재 속도에 대해 마찰을 먼저 적용해야 함.
            
            # Current Velocity (Prediction based)
            vx = px - old_x
            vy = py - old_y
            vz = pz - old_z
            
            # Normal Component of Velocity (v . n)
            v_dot_n = vx * nx + vy * ny + vz * nz
            
            # Tangential Velocity (v_t = v - v_n)
            vt_x = vx - v_dot_n * nx
            vt_y = vy - v_dot_n * ny
            vt_z = vz - v_dot_n * nz
            
            # Apply Friction Damping
            # scale = 1.0 (No friction) ~ 0.0 (Full stop)
            # Simple Damping: v_t_new = v_t * (1 - mu)
            f_scale = 1.0 - sphere_friction
            if f_scale < 0.0: f_scale = 0.0
            
            # Update Velocity (Position) with Friction
            # (Note: Normal component is kept as is, Projection will handle it)
            px = old_x + (v_dot_n * nx) + (vt_x * f_scale)
            py = old_y + (v_dot_n * ny) + (vt_y * f_scale)
            pz = old_z + (v_dot_n * nz) + (vt_z * f_scale)
            
            # -----------------------------------------------------
            # [Step 2] Projection (SDF Push)
            # -----------------------------------------------------
            # 마찰이 적용된 위치에서 밖으로 밀어냄
            px += nx * penetration
            py += ny * penetration
            pz += nz * penetration

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
def compute_curvature_kernel(pos, curvature_out, num_x, num_y):
    """
    [신규 추가] 이산 라플라시안을 이용한 곡률 근사 계산
    curvature = || p_avg - p_center ||
    """
    # 2D Grid 인덱싱 (Structured Grid의 이점 활용)
    ix, iy = cuda.grid(2)
    
    if ix >= num_x or iy >= num_y:
        return

    idx = iy * num_x + ix

    # 경계(Boundary)에 있는 파티클은 곡률 계산에서 제외하거나 0으로 처리
    if ix == 0 or ix == num_x - 1 or iy == 0 or iy == num_y - 1:
        curvature_out[idx] = 0.0
        return

    # 중심점 위치
    cx, cy, cz = pos[idx, 0], pos[idx, 1], pos[idx, 2]

    # 상하좌우 이웃 (Von Neumann Neighborhood)
    # Left (ix-1, iy)
    l_idx = iy * num_x + (ix - 1)
    lx, ly, lz = pos[l_idx, 0], pos[l_idx, 1], pos[l_idx, 2]

    # Right (ix+1, iy)
    r_idx = iy * num_x + (ix + 1)
    rx, ry, rz = pos[r_idx, 0], pos[r_idx, 1], pos[r_idx, 2]

    # Up (ix, iy-1)
    u_idx = (iy - 1) * num_x + ix
    ux, uy, uz = pos[u_idx, 0], pos[u_idx, 1], pos[u_idx, 2]

    # Down (ix, iy+1)
    d_idx = (iy + 1) * num_x + ix
    dx, dy, dz = pos[d_idx, 0], pos[d_idx, 1], pos[d_idx, 2]

    # 이웃들의 평균 위치 계산
    avg_x = (lx + rx + ux + dx) * 0.25
    avg_y = (ly + ry + uy + dy) * 0.25
    avg_z = (lz + rz + uz + dz) * 0.25

    # 라플라시안 벡터 (Average - Center)
    diff_x = avg_x - cx
    diff_y = avg_y - cy
    diff_z = avg_z - cz

    # 벡터의 크기(Magnitude)가 곧 곡률의 척도
    curv = math.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
    
    curvature_out[idx] = curv

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
    if curvature[idx] < curvature_threshold:
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