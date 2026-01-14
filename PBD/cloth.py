import torch
import numpy as np
from numba import cuda
from PBD.module import predict_position_kernel, update_velocity_kernel
from PBD.module import solve_distance_constraint_colored_kernel
from PBD.module import (find_cell_start_end_kernel, 
                        solve_self_collision_friction_kernel, apply_aerodynamics_kernel,
                        solve_environment_collision_kernel, clear_counter_kernel, compute_curvature_kernel,compute_hash_kernel_v2)
from PBD.coloring import compute_graph_coloring
import math
import time # <--- 필수 추가
HASH_TABLE_SIZE = 1000003  # 해시 테이블 크기 (충분히 크게)
CELL_SIZE = 0.1          # 격자 크기 (파티클 간격과 비슷하거나 약간 크게)

class ClothSimulator:
    def __init__(self, width, height, physical_width=10.0, dt=0.01, substeps=10):
        """
        [Resolution Independent Setup]
        width, height: 파티클 격자의 해상도 (개수)
        physical_width: 천의 실제 물리적 가로 길이 (미터)
        """
        self.dt = dt
        self.substeps = substeps
        
        # 1. Spacing 및 물리 크기 자동 계산
        # 해상도(width)가 변해도 물리적 크기(physical_width)는 유지됨
        self.num_x = width
        self.num_y = height
        num_particles = width * height
        self.num_particles = num_particles
        
        # 파티클 간격 = 물리적 길이 / (개수 - 1)
        spacing = physical_width / (width - 1)
        self.spacing = spacing
        
        # 물리적 세로 길이 (정사각형 격자 가정)
        physical_height = spacing * (height - 1)

        print(f"🎓 Simulation Init: Resolution=({width}x{height})")
        print(f"   -> Physical Size=({physical_width:.2f}m x {physical_height:.2f}m)")
        print(f"   -> Particle Spacing={spacing:.4f}m")

        # 2. Sphere 설정 (천의 크기에 비례하여 자동 조절)
        # 반지름: 천 가로 폭의 30%
        sphere_radius = physical_width * 0.3
        
        # 위치: 천의 정중앙(X, Z), 높이(Y)는 반지름의 절반만큼 아래로 내려서 윗면이 0 근처에 오게 함
        sphere_center_x = physical_width * 0.5 + sphere_radius
        sphere_center_y = sphere_radius + 0.3
        sphere_center_z = physical_height * 0.5
        
        sphere_center = np.array([sphere_center_x, sphere_center_y, sphere_center_z], dtype=np.float32)
        
        print(f"   -> Sphere: Center={sphere_center}, Radius={sphere_radius:.2f}")
        self.sphere_params = cuda.to_device(np.concatenate([sphere_center, [sphere_radius]]))

        # 마찰 및 바닥 설정
        self.floor_height = 0.0
        self.floor_friction = 0.9   # 바닥: 거침
        self.sphere_friction = 0.02 # 구체: 매우 미끄러움 (Sliding 유도)

        # 3. 파티클 초기 위치 생성 (Dropping Scenario)
        pos_host = np.zeros((num_particles, 3), dtype=np.float32)
        
        # 시작 높이: 구체 윗면보다 약간 위
        start_y = sphere_center_y + sphere_radius + (physical_width * 0.1)
        
        for y in range(height):
            for x in range(width):
                idx = y * width + x

                pos_x = x * spacing
                pos_z = y * spacing
                pos_y = start_y 
                
                # 미세한 노이즈로 자연스러운 주름 유도
                pos_y += np.random.uniform(-spacing*0.1, spacing*0.1)

                pos_host[idx] = [pos_x, pos_y, pos_z]

        # 4. 제약 조건 (Constraints) 생성: Structural + Shear
        constraints = []
        rest_lengths_list = []

        for y in range(height):
            for x in range(width):
                idx = y * width + x
                
                # (1) Structural (가로/세로)
                if x < width - 1: 
                    constraints.append([idx, idx + 1])
                    rest_lengths_list.append(spacing)
                if y < height - 1: 
                    constraints.append([idx, idx + width])
                    rest_lengths_list.append(spacing)
                
                # (2) Shear (대각선) - 천의 뒤틀림 방지
                diag_dist = spacing * math.sqrt(2)
                if x < width - 1 and y < height - 1:
                    constraints.append([idx, idx + width + 1])      # ↘
                    rest_lengths_list.append(diag_dist)
                    constraints.append([idx + 1, idx + width])      # ↙
                    rest_lengths_list.append(diag_dist)

        self.num_constraints = len(constraints)
        print(f"   -> Constraints Generated: {self.num_constraints} (Structural + Shear)")

        # 5. Graph Coloring (병렬 처리를 위한 배치 분할)
        print("   -> Computing Graph Coloring...")
        color_batches_host = compute_graph_coloring(num_particles, constraints)
        
        # 배치들을 GPU로 업로드
        self.d_color_batches = []
        for batch in color_batches_host:
            self.d_color_batches.append(cuda.to_device(batch))
        print(f"   -> Graph Coloring Done: {len(self.d_color_batches)} batches.")

        # 6. 데이터 GPU 할당
        self.d_pos = cuda.to_device(pos_host)
        self.d_pos_pred = cuda.to_device(pos_host)
        self.d_vel = cuda.to_device(np.zeros_like(pos_host))

        # [Mass Scaling] 해상도에 따른 질량 자동 조절
        # 기준: spacing=0.1일 때 mass=1.0
        ref_spacing = 0.1
        particle_mass = (spacing / ref_spacing) ** 2 
        mass_inv = np.ones(num_particles, dtype=np.float32) * (1.0 / particle_mass)
        self.d_mass_inv = cuda.to_device(mass_inv)

        # 제약 조건 데이터 업로드
        self.d_constraints = cuda.to_device(np.array(constraints, dtype=np.int32))
        self.d_rest_lengths = cuda.to_device(np.array(rest_lengths_list, dtype=np.float32))

        # 7. Spatial Hashing 및 충돌 관련 버퍼
        self.d_particle_hashes = cuda.device_array(self.num_particles, dtype=np.int32)
        self.d_particle_indices = cuda.device_array(self.num_particles, dtype=np.int32)
        
        self.d_cell_start = cuda.device_array(HASH_TABLE_SIZE, dtype=np.int32)
        self.d_cell_end = cuda.device_array(HASH_TABLE_SIZE, dtype=np.int32)
        
        # Self-Collision 파라미터
        self.thickness = spacing * 0.7
        self.collision_margin = self.thickness * 0.5
        
        # 디버깅/시각화용 Penetration 버퍼
        self.d_penetration = cuda.device_array(self.num_particles, dtype=np.float32)

        # 8. 렌더링용 Topology (Faces) 생성
        faces_list = []
        for y in range(height - 1):
            for x in range(width - 1):
                idx = y * width + x
                # Triangle 1
                faces_list.append([idx, idx + 1, idx + width + 1])
                # Triangle 2
                faces_list.append([idx, idx + width + 1, idx + width])
        
        faces_array = np.array(faces_list, dtype=np.int32)
        self.num_faces = len(faces_array)
        self.d_faces = cuda.to_device(faces_array)

        # [NEW] View-Dependent Culling을 위한 데이터 할당
        # 1. 법선 벡터 버퍼 (N, 3)
        self.frame_count = 0
        self.d_normals = cuda.device_array((self.num_particles, 3), dtype=np.float32)
        # 2. 가시성 점수 버퍼 (N,)
        self.d_visibility = cuda.device_array(self.num_particles, dtype=np.float32)
        
        # 3. 가상 카메라 위치 설정
        # 렌더링 시점과 비슷하게 맞춰주면 효과가 좋습니다.
        # (예: 약간 위에서 아래로 내려다보는 시점)
        camera_pos_host = np.array([self.num_x * self.spacing * 0.5,  # X: 천 중앙
                                    10.0,                             # Y: 높게
                                    self.num_y * self.spacing * 1.5], # Z: 천 앞쪽으로 멀리
                                   dtype=np.float32)
        self.d_camera_pos = cuda.to_device(camera_pos_host)
        print(f"   -> Virtual Camera Pos for Culling: {camera_pos_host}")

        # 9. 공기 역학 (Aerodynamics)
        self.rho = 1.225
        self.drag_coeff = 2.5
        self.lift_coeff = 0.5
        # Z방향으로 부는 바람
        self.wind_vel = cuda.to_device(np.array([0.0, 0.0, 3.0], dtype=np.float32))

        # 10. CUDA 실행 설정
        self.threads_per_block = 256
        self.blocks_particles = (self.num_particles + self.threads_per_block - 1) // self.threads_per_block

        self.threads_per_block_2d = (16, 16)
        self.blocks_per_grid_x = int(np.ceil(self.num_x / 16))
        self.blocks_per_grid_y = int(np.ceil(self.num_y / 16))
        self.blocks_per_grid_2d = (self.blocks_per_grid_x, self.blocks_per_grid_y)

        self.d_debug_skip_count = cuda.to_device(np.array([0], dtype=np.int32))
        self.last_sort_time = 0.0 # [NEW] 마지막 프레임의 정렬 시간 저장

        # 곡률 기반 컬링
        self.d_curvature = cuda.device_array(self.num_particles, dtype=np.float32)
        self.curvature_threshold = 0.002

# ---------------------------------------------------------
        # [GPU Memory Usage Info] 초기화 후 메모리 사용량 출력
        # ---------------------------------------------------------
        try:
            # [수정됨] Numba 컨텍스트를 통해 메모리 정보 가져오기
            # cuda.current_context().get_memory_info()는 (free, total) 튜플을 바이트 단위로 반환합니다.
            # 이 방식이 더 호환성이 높습니다.
            ctx = cuda.current_context()
            free_mem, total_mem = ctx.get_memory_info()
            
            used_mem = total_mem - free_mem
            
            # GB 단위로 변환
            total_gb = total_mem / (1024**3)
            used_gb = used_mem / (1024**3)
            free_gb = free_mem / (1024**3)
            usage_percent = (used_mem / total_mem) * 100 if total_mem > 0 else 0

            print("-" * 50)
            print(f"💾 GPU Memory Usage (After Init):")
            print(f"   - Total: {total_gb:.2f} GB")
            print(f"   - Used : {used_gb:.2f} GB ({usage_percent:.1f}%)")
            print(f"   - Free : {free_gb:.2f} GB")
            print("-" * 50)
            
        except Exception as e:
            # 만약 이 방식도 실패하면 에러 메시지를 자세히 출력
            import traceback
            traceback.print_exc()
            print(f"[Warning] Failed to get GPU memory info: {e}")

    def _sort_particles_torch(self):
        # 1. [Sync Numba] 시작 전 대기
        cuda.synchronize()

        # 타이머 시작 (여기서부터 정렬 단계로 간주)
        t_start = time.perf_counter() 

        # --- PyTorch 영역 ---
        hashes_torch = torch.as_tensor(self.d_particle_hashes, device='cuda')
        indices_torch = torch.as_tensor(self.d_particle_indices, device='cuda')
        
        sorted_indices = torch.argsort(hashes_torch, descending=False, stable=True)
        
        hashes_sorted = hashes_torch[sorted_indices].contiguous()
        indices_sorted = indices_torch[sorted_indices].contiguous()
        
        # [Sync PyTorch] PyTorch 작업 완료 대기
        torch.cuda.synchronize()
        # --- PyTorch 영역 끝 ---

        # 3. PyTorch -> Numba (데이터 덮어쓰기)
        # 이 작업들도 정렬 시간에 포함되어야 합니다.
        sorted_hashes_cuda = cuda.as_cuda_array(hashes_sorted)
        sorted_indices_cuda = cuda.as_cuda_array(indices_sorted)
        
        # 비동기 복사 명령 내림
        self.d_particle_hashes.copy_to_device(sorted_hashes_cuda)
        self.d_particle_indices.copy_to_device(sorted_indices_cuda)

        # ==================================================================
        # [핵심 수정] 마지막 Numba 복사 작업이 끝날 때까지 기다려야 합니다.
        # ==================================================================
        cuda.synchronize() # Numba 스트림 완료 대기

        # 타이머 종료 (모든 데이터 이동이 완료된 후 측정)
        t_end = time.perf_counter()

        # 걸린 시간 저장 (단위: 초)
        self.last_sort_time = t_end - t_start

    def step(self):
        # clear_counter_kernel[1, 1](self.d_debug_skip_count)
        target_compliance = 0.0       # 완전 딱딱함 (비신축성, 실크/면)
        # target_compliance = 0.00001   # 아주 약간 늘어남 (나일론)
        # target_compliance = 0.005     # 고무줄/스판덱스
        # target_compliance = 0.0000001 # 거의 0에 가깝게 설정하여 기존 천 느낌 유지
        self.frame_count += 1

        dt_sub = self.dt / self.substeps

        for _ in range(self.substeps):
                
            # ------------------------------------------------------------------
            # [Stage 0] External Forces (Aerodynamics)
            # ------------------------------------------------------------------
            # 바람에 의한 양력/항력을 계산하여 현재 속도(vel)에 선반영
            blocks_faces = (self.num_faces + self.threads_per_block - 1) // self.threads_per_block
            
            apply_aerodynamics_kernel[blocks_faces, self.threads_per_block](
                self.d_pos,         # Position (for Normal calc)
                self.d_vel,         # Velocity (Force applied here)
                self.d_faces,       # Topology
                self.wind_vel,      # Wind Vector
                self.rho,           # Air Density
                self.drag_coeff,    # Cd
                self.lift_coeff,    # Cl
                dt_sub,             # Time step
                self.num_faces
            )

            # ------------------------------------------------------------------
            # [Stage 1] Prediction & Integration
            # ------------------------------------------------------------------
            # 중력 적용 및 위치 예측 (Explicit Integration)
            predict_position_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos, self.d_vel, self.d_pos_pred, self.d_mass_inv, 
                dt_sub, -9.8, self.num_particles
            )

            # ------------------------------------------------------------------
            # [Stage 1.5] Environment Collision (Sphere SDF)
            # ------------------------------------------------------------------
            # 예측된 위치가 구 안으로 들어갔다면 즉시 밀어냄
            solve_environment_collision_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos_pred,    
                self.d_pos,         
                self.d_mass_inv,
                self.sphere_params,
                self.sphere_friction, # 구 마찰
                self.floor_height,    # [NEW] 바닥 높이
                self.floor_friction,  # [NEW] 바닥 마찰
                dt_sub,
                self.num_particles,
                self.collision_margin
            )
            # ------------------------------------------------------------------
            # [Stage 2] Distance Constraints (XPBD + Graph Coloring)
            # ------------------------------------------------------------------
            # Graph Coloring으로 병렬화된 거리 제약 조건 해결
            for d_batch in self.d_color_batches:
                blocks_batch = (d_batch.shape[0] + self.threads_per_block - 1) // self.threads_per_block

                solve_distance_constraint_colored_kernel[blocks_batch, self.threads_per_block](
                    self.d_pos_pred, self.d_mass_inv, self.d_constraints, self.d_rest_lengths,
                    d_batch, dt_sub, target_compliance
                )

            # ------------------------------------------------------------------
            # [Stage 3] Self-Collision (Spatial Hashing)
            # ------------------------------------------------------------------
            # 3-1. Reset Grid
            self.d_cell_start[:] = -1 
            self.d_cell_end[:] = -1
            self.d_penetration[:] = 0.0

            threads_1d = 256
            blocks_1d = int(math.ceil(self.num_particles / threads_1d))

            compute_curvature_kernel[self.blocks_per_grid_2d, self.threads_per_block_2d](
                self.d_pos, 
                self.d_curvature, 
                self.num_x, 
                self.num_y
            )

            # 3-2. Compute Hash
            compute_hash_kernel_v2[blocks_1d, threads_1d](
                self.d_pos_pred,             # 1. 위치
                self.d_particle_hashes,      # 2. 해시값 저장소 (이름 수정됨!)
                self.d_particle_indices,     # 3. [추가됨] 파티클 ID 저장소 (정렬용)
                self.d_cell_start,           # 4. (Dummy)
                self.d_cell_end,             # 5. (Dummy)
                self.d_curvature,            # 6. 곡률
                self.curvature_threshold,    # 7. 임계값
                self.num_particles,          # 8. 파티클 수
                CELL_SIZE,                   # 9. 셀 크기
                HASH_TABLE_SIZE              # 10. 해시 테이블 크기
            )

            # 3-3. Sort Particles (PyTorch Radix Sort - Zero Copy)
            self._sort_particles_torch()

            # 3-4. Find Cell Bounds
            find_cell_start_end_kernel[self.blocks_particles, self.threads_per_block](
                self.d_particle_hashes, self.d_cell_start, self.d_cell_end, self.num_particles
            )

            # 3-5. Solve Collision with Friction
            solve_self_collision_friction_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos_pred,        # Candidate Position
                self.d_pos,             # Previous Position (for relative velocity)
                self.d_mass_inv, 
                self.d_cell_start, self.d_cell_end, 
                self.d_particle_indices, self.d_particle_hashes, 
                self.num_particles, self.thickness, self.d_penetration,
                dt_sub,
                self.d_visibility, self.frame_count,
                self.d_debug_skip_count
            )

            # ------------------------------------------------------------------
            # [Stage 4] Velocity Update
            # ------------------------------------------------------------------
            # 위치 확정 및 속도 갱신 (Damping 포함)
            update_velocity_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos, self.d_vel, self.d_pos_pred, dt_sub, self.num_particles
            )

        # skipped = self.d_debug_skip_count.copy_to_host()[0]
        # total = self.num_particles * self.substeps
        # print(f"   [DEBUG] Skipped Collisions: {skipped} / {total} ({skipped/total*100:.1f}%)")

    def get_positions(self):
        return self.d_pos.copy_to_host()

    def get_penetration_depth(self):
        return self.d_penetration.copy_to_host()

    def get_velocities(self):
        return self.d_vel.copy_to_host()

    def get_compression_feature(self, positions=None):
        """
        [핵심 기능] 기하학적 정보(Geometry Feature) 추출
        각 파티클 주변의 스프링이 얼마나 압축되었는지 계산 (Strain)
        Output: (N, 1) float32 array
           - 값 < 1.0: 압축됨 (주름짐, 충돌 위험 높음)
           - 값 >= 1.0: 팽팽함 (안전함)
        """
        if positions is None:
            positions = self.get_positions() # (N, 3)

        # 2D 그리드 형태로 변환
        pos_grid = positions.reshape(self.num_y, self.num_x, 3)
        
        # 결과 담을 배열
        strain_map = np.zeros((self.num_y, self.num_x), dtype=np.float32)
        
        # 1. 가로 방향 (Horizontal) 변형률
        # diff_h: (H, W-1, 3)
        diff_h = pos_grid[:, 1:] - pos_grid[:, :-1]
        dist_h = np.linalg.norm(diff_h, axis=2)
        ratio_h = dist_h / self.spacing # Rest Length(spacing) 대비 비율
        
        # 2. 세로 방향 (Vertical) 변형률
        # diff_v: (H-1, W, 3)
        diff_v = pos_grid[1:, :] - pos_grid[:-1, :]
        dist_v = np.linalg.norm(diff_v, axis=2)
        ratio_v = dist_v / self.spacing

        # 3. 각 파티클에 할당 (Average Strain)
        # 파티클 입장에서 자신에게 연결된 스프링들의 평균 비율을 구함
        
        # 왼쪽/오른쪽 스프링 더하기
        strain_accum = np.zeros((self.num_y, self.num_x), dtype=np.float32)
        count_accum = np.zeros((self.num_y, self.num_x), dtype=np.float32)
        
        # Horizontal Contributions
        strain_accum[:, :-1] += ratio_h # Right connection for (x, y)
        count_accum[:, :-1] += 1
        strain_accum[:, 1:] += ratio_h  # Left connection for (x+1, y)
        count_accum[:, 1:] += 1
        
        # Vertical Contributions
        strain_accum[:-1, :] += ratio_v # Bottom connection
        count_accum[:-1, :] += 1
        strain_accum[1:, :] += ratio_v  # Top connection
        count_accum[1:, :] += 1
        
        # 평균 계산 (0으로 나누기 방지 위해 eps 추가 가능하지만 여기선 count가 무조건 >=2)
        avg_strain = strain_accum / count_accum
        
        # (N, 1) 형태로 Flatten
        return avg_strain.reshape(-1, 1)