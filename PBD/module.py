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
def solve_self_collision_friction_kernel(pos_pred, pos_old, mass_inv, 
                                         cell_start, cell_end, 
                                         sorted_indices, particle_hashes, 
                                         num_particles, thickness, 
                                         penetration_buffer, dt):
    """
    [Professor's Final Version]
    - Spatial Hashing 기반 Self-Collision
    - Stiffness 적용 (Soft Constraint)
    - Max Displacement Clamp 적용 (Explosion 방지)
    - Coulomb Friction 적용
    """
    idx = cuda.grid(1)
    if idx >= num_particles: return

    w_i = mass_inv[idx]
    if w_i == 0.0: return
    
    # 1. 현재 예측 위치 (Candidate Position)
    px = pos_pred[idx, 0]
    py = pos_pred[idx, 1]
    pz = pos_pred[idx, 2]

    # 2. 이전 위치 (Previous Position) - 상대 속도 및 마찰 계산용
    px_old = pos_old[idx, 0]
    py_old = pos_old[idx, 1]
    pz_old = pos_old[idx, 2]
    
    # Spatial Hashing 좌표 계산 (Hardcoded CELL_SIZE=0.1 주의)
    # 만약 spacing이 바뀌었다면 이 0.1도 맞춰서 바꿔줘야 함 (보통 spacing과 같거나 조금 크게)
    cell_size = 0.1
    grid_x = int(math.floor(px / cell_size))
    grid_y = int(math.floor(py / cell_size))
    grid_z = int(math.floor(pz / cell_size))
    
    # --- Tuning Parameters ---
    friction_mu_s = 0.5   # 정지 마찰 계수
    friction_mu_k = 0.3   # 운동 마찰 계수
    
    # [핵심 1] Stiffness: 한 번에 100% 밀어내지 않고 부드럽게 (10%씩)
    # Substeps가 많으므로(30~40회), 작은 값이어도 충분히 밀어냄
    collision_stiffness = 0.1 
    
    # [핵심 2] Max Displacement: 한 번에 이동 가능한 최대 거리 제한
    # 폭발적인 힘이 발생해도 이 이상 움직이지 못하게 막음 (Safety Clamp)
    max_correction = thickness * 0.1

    max_depth = 0.0 # 시각화/디버깅용

    # 3x3x3 이웃 셀 탐색
    for z in range(-1, 2):
        for y in range(-1, 2):
            for x in range(-1, 2):
                neighbor_x = grid_x + x
                neighbor_y = grid_y + y
                neighbor_z = grid_z + z
                
                # Hash Function
                h = (neighbor_x * 73856093) ^ (neighbor_y * 19349663) ^ (neighbor_z * 83492791)
                h = h % 1000003 # HASH_TABLE_SIZE
                
                start_idx = cell_start[h]
                end_idx = cell_end[h]
                
                if start_idx == -1: continue 

                for k in range(start_idx, end_idx):
                    j = sorted_indices[k]
                    if idx == j: continue 
                    
                    # 이웃 파티클 j 정보 로드
                    jx = pos_pred[j, 0]
                    jy = pos_pred[j, 1]
                    jz = pos_pred[j, 2]
                    
                    w_j = mass_inv[j]
                    w_sum = w_i + w_j
                    if w_sum == 0.0: continue

                    # --- Collision Detection ---
                    dx = px - jx
                    dy = py - jy
                    dz = pz - jz
                    
                    dist_sq = dx*dx + dy*dy + dz*dz
                    min_dist = thickness * 2.0 # 양쪽 반지름의 합
                    
                    # 충돌 판정
                    if dist_sq < (min_dist * min_dist) and dist_sq > 1e-12:
                        dist = math.sqrt(dist_sq)
                        penetration = min_dist - dist
                        
                        # 시각화용 데이터 기록
                        if penetration > max_depth:
                            max_depth = penetration
                        
                        # Normal Vector (j -> i 방향, 즉 i를 밀어내는 방향)
                        nx = dx / dist
                        ny = dy / dist
                        nz = dz / dist
                        
                        # ------------------------------------------------
                        # [Step 1] Normal Position Correction (Repulsion)
                        # ------------------------------------------------
                        # correction magnitude (scalar)
                        correction_mag = (penetration / w_sum) * collision_stiffness
                        
                        # [Safety] Clamp Correction Force
                        if correction_mag > max_correction:
                            correction_mag = max_correction

                        # Apply Position Delta (Atomic Add)
                        dx_n = nx * correction_mag * w_i
                        dy_n = ny * correction_mag * w_i
                        dz_n = nz * correction_mag * w_i
                        
                        cuda.atomic.add(pos_pred, (idx, 0), dx_n)
                        cuda.atomic.add(pos_pred, (idx, 1), dy_n)
                        cuda.atomic.add(pos_pred, (idx, 2), dz_n)
                        
                        # ------------------------------------------------
                        # [Step 2] Friction Correction (Tangential)
                        # ------------------------------------------------
                        # i와 j의 상대 변위(Relative Displacement) 계산
                        jx_old = pos_old[j, 0]
                        jy_old = pos_old[j, 1]
                        jz_old = pos_old[j, 2]
                        
                        # Delta P_i
                        disp_i_x = (px - px_old)
                        disp_i_y = (py - py_old)
                        disp_i_z = (pz - pz_old)
                        
                        # Delta P_j
                        disp_j_x = (jx - jx_old)
                        disp_j_y = (jy - jy_old)
                        disp_j_z = (jz - jz_old)
                        
                        # Relative Displacement
                        rel_disp_x = disp_i_x - disp_j_x
                        rel_disp_y = disp_i_y - disp_j_y
                        rel_disp_z = disp_i_z - disp_j_z
                        
                        # Tangential Component 추출
                        # v_t = v_rel - (v_rel . n) * n
                        dot_n = rel_disp_x * nx + rel_disp_y * ny + rel_disp_z * nz
                        
                        lat_x = rel_disp_x - dot_n * nx
                        lat_y = rel_disp_y - dot_n * ny
                        lat_z = rel_disp_z - dot_n * nz
                        
                        lat_len = math.sqrt(lat_x*lat_x + lat_y*lat_y + lat_z*lat_z)
                        
                        if lat_len > 1e-10:
                            # Tangent Direction
                            tx = lat_x / lat_len
                            ty = lat_y / lat_len
                            tz = lat_z / lat_len
                            
                            # Coulomb Friction Constraint
                            # 수직 항력(Impulse)에 비례하는 한계치 설정
                            # delta_lambda_n = correction_mag (위에서 계산한 값)
                            
                            # 마찰력 한계 (Kinetic Limit)
                            limit = friction_mu_k * correction_mag * w_sum # w_sum을 곱해 위치 스케일 보정
                            
                            friction_correction = 0.0
                            
                            # Static vs Kinetic 판별
                            if lat_len < (friction_mu_s * correction_mag * w_sum):
                                # Static: 움직임을 완전히 상쇄
                                friction_correction = lat_len
                            else:
                                # Kinetic: 한계치만큼만 저항
                                friction_correction = limit
                            
                            # Apply Friction Delta (반대 방향으로 이동)
                            scale = friction_correction / w_sum
                            
                            cuda.atomic.add(pos_pred, (idx, 0), -tx * scale * w_i)
                            cuda.atomic.add(pos_pred, (idx, 1), -ty * scale * w_i)
                            cuda.atomic.add(pos_pred, (idx, 2), -tz * scale * w_i)

    # 루프 종료 후 최대 침투 깊이 기록
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
                                       dt, num_particles):
    """
    [Environment Solver] Sphere + Floor (Ground Plane)
    한 커널에서 두 가지 환경 충돌을 모두 처리함.
    """
    idx = cuda.grid(1)
    if idx < num_particles:
        w = mass_inv[idx]
        if w == 0.0: return 

        # 1. 위치 로드
        px = pos_pred[idx, 0]
        py = pos_pred[idx, 1]
        pz = pos_pred[idx, 2]

        old_x = pos_old[idx, 0]
        old_y = pos_old[idx, 1]
        old_z = pos_old[idx, 2]

        # ---------------------------------------------------------
        # [Object 1] Sphere SDF
        # ---------------------------------------------------------
        cx, cy, cz, radius = sphere_params[0], sphere_params[1], sphere_params[2], sphere_params[3]
        
        dx = px - cx
        dy = py - cy
        dz = pz - cz
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        # 충돌 플래그 및 Normal/Penetration 저장 변수
        collided = False
        nx, ny, nz = 0.0, 0.0, 0.0
        friction = 0.0

        if dist < radius and dist > 1e-6:
            # 구체 충돌 발생
            collided = True
            penetration = radius - dist
            nx, ny, nz = dx/dist, dy/dist, dz/dist
            friction = sphere_friction
            
            # 위치 보정 (Projection)
            px += nx * penetration
            py += ny * penetration
            pz += nz * penetration

        # ---------------------------------------------------------
        # [Object 2] Floor Plane (y > floor_height)
        # ---------------------------------------------------------
        # 구체 처리가 끝난 px, py, pz를 기준으로 바닥 검사
        # (구체에서 밀려나왔더니 바닥을 뚫는 경우도 처리됨)
        
        if py < floor_height:
            # 바닥 충돌 발생
            # 만약 구체와 동시에 충돌했다면? -> 바닥이 우선순위가 높음 (Hard Floor)
            # 혹은 벡터를 합쳐야 하지만, 여기서는 순차적으로 처리 (Sequential Impulse)
            
            collided = True
            penetration = floor_height - py
            
            # 바닥의 Normal은 무조건 (0, 1, 0)
            # 바닥 마찰 계수 적용
            friction = floor_friction 
            nx, ny, nz = 0.0, 1.0, 0.0 
            
            # 위치 보정 (Projection)
            py = floor_height # 강제로 바닥 위로 올림
            
            # 갱신된 좌표 반영
            px = px
            pz = pz

        # ---------------------------------------------------------
        # [Common] Friction Apply
        # ---------------------------------------------------------
        if collided:
            # 위치가 보정되었으므로(px, py, pz), 마찰 적용
            # 이동 벡터 (Delta)
            delta_x = px - old_x
            delta_y = py - old_y
            delta_z = pz - old_z
            
            # Normal Component (v . n)
            dot_n = delta_x * nx + delta_y * ny + delta_z * nz
            
            # Tangent Component
            tan_x = delta_x - dot_n * nx
            tan_y = delta_y - dot_n * ny
            tan_z = delta_z - dot_n * nz
            
            # Friction Damping
            scale = 1.0 - friction
            if scale < 0.0: scale = 0.0
            
            # Apply Friction
            px = old_x + (delta_x * nx * nx) + (tan_x * scale)
            py = old_y + (delta_y * ny * ny) + (tan_y * scale)
            pz = old_z + (delta_z * nz * nz) + (tan_z * scale)

        # 최종 위치 저장
        pos_pred[idx, 0] = px
        pos_pred[idx, 1] = py
        pos_pred[idx, 2] = pz