import torch
import numpy as np
from numba import cuda
from PBD.module import predict_position_kernel, update_velocity_kernel
from PBD.module import solve_distance_constraint_colored_kernel
from PBD.module import compute_hash_kernel, find_cell_start_end_kernel, solve_self_collision_friction_kernel, apply_aerodynamics_kernel,solve_environment_collision_kernel
from PBD.coloring import compute_graph_coloring
import math

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

        # 9. 공기 역학 (Aerodynamics)
        self.rho = 1.225
        self.drag_coeff = 2.5
        self.lift_coeff = 0.5
        # Z방향으로 부는 바람
        self.wind_vel = cuda.to_device(np.array([0.0, 0.0, 3.0], dtype=np.float32))

        # 10. CUDA 실행 설정
        self.threads_per_block = 256
        self.blocks_particles = (self.num_particles + self.threads_per_block - 1) // self.threads_per_block

    def _sort_particles_torch(self):
        """
        [Professor's Refinement] PyTorch를 이용한 고속 정렬
        Numba Device Array <-> PyTorch Tensor 간의 Zero-Copy 변환을 수행합니다.
        """
        # 1. Numba -> PyTorch (Zero-Copy View 생성)
        # __cuda_array_interface__를 통해 GPU 메모리 주소를 공유합니다.
        ctx = cuda.current_context()
        
        # Numba 배열의 포인터를 PyTorch가 이해하도록 변환
        hashes_torch = torch.as_tensor(self.d_particle_hashes, device='cuda')
        indices_torch = torch.as_tensor(self.d_particle_indices, device='cuda')
        
        # 2. PyTorch의 강력한 Radix Sort 수행
        # stable=True는 같은 해시값을 가진 파티클들의 순서를 유지해줌 (물리적 안정성에 도움)
        sorted_indices = torch.argsort(hashes_torch, descending=False, stable=True)
        
        # 3. 정렬된 결과로 재배치 (Fancy Indexing - 여기서 VRAM 복사 발생)
        # 하지만 C++ 레벨에서 최적화된 커널이 돌기 때문에 매우 빠름
        hashes_sorted = hashes_torch[sorted_indices]
        indices_sorted = indices_torch[sorted_indices]
        
        # 4. PyTorch -> Numba (데이터 덮어쓰기)
        # PyTorch 텐서의 __cuda_array_interface__를 Numba가 읽어서 복사
        # contiguous()는 메모리가 연속적인지 확인하는 안전장치
        
        # 방법 A: copy_to_device (명시적)
        # self.d_particle_hashes.copy_to_device(cuda.as_cuda_array(hashes_sorted))
        
        # 방법 B: Direct Copy (추천)
        # Numba device array에 다른 CUDA array 인터페이스 객체를 넣으면 D2D 복사가 일어남
        
        # [주의] PyTorch 텐서를 바로 Numba array에 할당할 수 없으므로,
        # 아래와 같이 CUDA Interface를 통해 값을 복사해야 함.
        
        # 4-1. PyTorch Tensor -> Numba Device Array View
        sorted_hashes_cuda = cuda.as_cuda_array(hashes_sorted)
        sorted_indices_cuda = cuda.as_cuda_array(indices_sorted)
        
        # 4-2. 원본 버퍼에 복사 (Device to Device Copy)
        self.d_particle_hashes.copy_to_device(sorted_hashes_cuda)
        self.d_particle_indices.copy_to_device(sorted_indices_cuda)

    def step(self):

        target_compliance = 0.0       # 완전 딱딱함 (비신축성, 실크/면)
        # target_compliance = 0.00001   # 아주 약간 늘어남 (나일론)
        # target_compliance = 0.005     # 고무줄/스판덱스
        # target_compliance = 0.0000001 # 거의 0에 가깝게 설정하여 기존 천 느낌 유지

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

            # 3-2. Compute Hash
            compute_hash_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos_pred, self.d_particle_hashes, self.d_particle_indices, self.num_particles
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
                dt_sub
            )

            # ------------------------------------------------------------------
            # [Stage 4] Velocity Update
            # ------------------------------------------------------------------
            # 위치 확정 및 속도 갱신 (Damping 포함)
            update_velocity_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos, self.d_vel, self.d_pos_pred, dt_sub, self.num_particles
            )

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