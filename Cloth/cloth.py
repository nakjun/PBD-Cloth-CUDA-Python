import torch
import numpy as np
from numba import cuda
from PBD.module import predict_position_kernel, update_velocity_kernel
from PBD.module import solve_distance_constraint_colored_kernel
from PBD.module import compute_hash_kernel, find_cell_start_end_kernel, solve_self_collision_friction_kernel, apply_aerodynamics_kernel
from PBD.coloring import compute_graph_coloring
import math

HASH_TABLE_SIZE = 1000003  # 해시 테이블 크기 (충분히 크게)
CELL_SIZE = 0.1          # 격자 크기 (파티클 간격과 비슷하거나 약간 크게)

class ClothSimulator:
    def __init__(self, width, height, spacing=0.1):
        self.dt = 0.01
        self.substeps = 10  # PBD는 작은 스텝을 여러 번 돌려야 안정적임

        # 1. 그리드 메쉬 생성
        self.num_x = width
        self.num_y = height
        num_particles = width * height

        # Flag 형태의 천을 생성합니다.
        pos_host = np.zeros((num_particles, 3), dtype=np.float32)
        start_y = 2.0 # 높이 조금 Up (깃발 느낌)
        flag_wave_amplitude = spacing * 0.6  # 파도 높이 (조절 가능)
        flag_wave_frequency = 2.0            # 파도 빈도 (파장=늘릴수록 느리고 큼)
        flag_offset = 0.0                    # y방향 오프셋

        for y in range(height):
            for x in range(width):
                idx = y * width + x

                # 깃발은 X=0 쪽 (막대기)에서 시작해서 +X로 뻗음, Z방향으로 물결침
                pos_x = x * spacing
                pos_y = (height - y - 1) * spacing + start_y  # 맨 위가 start_y (축 방향 보정)
                pos_z = math.sin(x * flag_wave_frequency * math.pi / width) * flag_wave_amplitude
                # "날리는" 느낌 가미: 아래로 갈수록 파도 줄어듦
                pos_z *= (1.0 - y / (height-1)) if height > 1 else 1.0

                pos_host[idx] = [pos_x, pos_y, pos_z]

        indices = []
        constraints = []

        # [SCENE SETUP] "The Squeezed Curtain"
        # 윗부분을 30%로 압축하여 강제로 주름을 만듭니다.
        compression_ratio = 0.5  # 1.0이면 평평함, 0.3이면 매우 쭈글쭈글함

        # for y in range(height):
        #     for x in range(width):
        #         idx = y * width + x

        #         # 아코디언/커튼 효과: x축은 압축, z축은 사인파, 아래쪽이 펼쳐지게 y로 lerp할 수도 있음
        #         center_x = (width - 1) * spacing / 2.0
        #         original_x = x * spacing

        #         # 주름진 초기 상태: sine wave를 z축에 덧입힘
        #         freq = 1.5  # 주름의 빈도
        #         amp = spacing * 2.0 # 주름의 깊이

        #         z_offset = np.sin(x * freq) * amp

        #         # X축을 compression_ratio로 완전히 압축 (더 복잡하게 하려면 y에 따라 lerp 가능)
        #         pos_host[idx] = [x * spacing * compression_ratio, -y * spacing + (height * spacing), z_offset]

        #         # Structural Constraints 생성 (동일)
        #         if x < width - 1:
        #             constraints.append([idx, idx + 1])
        #         if y < height - 1:
        #             constraints.append([idx, idx + width])

        constraints = []
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                
                # 1. Structural (가로/세로) - 기존
                if x < width - 1: 
                    constraints.append([idx, idx + 1])
                if y < height - 1: 
                    constraints.append([idx, idx + width])
                
                # 2. Shear (대각선) - [NEW] 추가!
                # 천의 뒤틀림을 막아주어 형태를 유지함
                if x < width - 1 and y < height - 1:
                    constraints.append([idx, idx + width + 1])      # ↘ 대각선
                    constraints.append([idx + 1, idx + width])      # ↙ 대각선

        self.num_particles = num_particles
        self.num_constraints = len(constraints)

        # [NEW] Graph Coloring 수행 (CPU)
        print("Computing Graph Coloring...")
        color_batches_host = compute_graph_coloring(num_particles, constraints)

        # [NEW] 배치들을 GPU로 업로드
        self.d_color_batches = []
        for batch in color_batches_host:
            self.d_color_batches.append(cuda.to_device(batch))

        # 2. 데이터 GPU 할당
        self.d_pos = cuda.to_device(pos_host)
        self.d_pos_pred = cuda.to_device(pos_host)  # 예측 위치 버퍼
        self.d_vel = cuda.to_device(np.zeros_like(pos_host))

        # [질량 설정 수정]
        # 맨 윗줄 전체(y=0)를 고정 (커튼 연출)
        mass_inv = np.ones(num_particles, dtype=np.float32)
        # for x in range(width):
        #     mass_inv[x] = 0.0 
        mass_inv[0] = 0.0
        self.d_mass_inv = cuda.to_device(mass_inv)

        # 제약 조건 GPU 할당
        constraints = np.array(constraints, dtype=np.int32)
        self.d_constraints = cuda.to_device(constraints)

        # Rest Length 계산
        # rest_lengths = np.linalg.norm(pos_host[constraints[:, 0]] - pos_host[constraints[:, 1]], axis=1)
        rest_lengths_list = []
        for y in range(height):
            for x in range(width):
                # Structural
                if x < width - 1: rest_lengths_list.append(spacing)
                if y < height - 1: rest_lengths_list.append(spacing)
                # Shear
                if x < width - 1 and y < height - 1:
                    rest_lengths_list.append(spacing * math.sqrt(2)) # 대각선 길이
                    rest_lengths_list.append(spacing * math.sqrt(2))

        rest_lengths = np.array(rest_lengths_list, dtype=np.float32)

        self.d_rest_lengths = cuda.to_device(rest_lengths.astype(np.float32))

        # CUDA Block/Grid 설정
        self.threads_per_block = 256
        self.blocks_particles = (self.num_particles + self.threads_per_block - 1) // self.threads_per_block
        self.blocks_constraints = (self.num_constraints + self.threads_per_block - 1) // self.threads_per_block

        # Self-Collision용 버퍼
        self.d_particle_hashes = cuda.device_array(self.num_particles, dtype=np.int32)
        self.d_particle_indices = cuda.device_array(self.num_particles, dtype=np.int32)
        
        # Grid Cell 정보 (Start/End)
        self.d_cell_start = cuda.device_array(HASH_TABLE_SIZE, dtype=np.int32)
        self.d_cell_end = cuda.device_array(HASH_TABLE_SIZE, dtype=np.int32)
        
        # 파티클 두께 (Self Collision 거리)
        self.thickness = spacing * 0.3 # 간격보다 조금 작게

        self.spacing = spacing

        # Penetration Depth 버퍼
        self.d_penetration = cuda.device_array(self.num_particles, dtype=np.float32)

        faces_list = []
        for y in range(height - 1):
            for x in range(width - 1):
                idx = y * width + x
                # Triangle 1: (idx, idx+1, idx+width+1)
                # Triangle 2: (idx, idx+width+1, idx+width)
                # 주의: 렌더링용과 다르게 '3' 같은 헤더 없이 순수 인덱스만 저장
                faces_list.append([idx, idx + 1, idx + width + 1])
                faces_list.append([idx, idx + width + 1, idx + width])

        faces_array = np.array(faces_list, dtype=np.int32)
        self.num_faces = len(faces_array)
        self.d_faces = cuda.to_device(faces_array) # GPU 업로드

        # 2. 공기 역학 파라미터 정의
        self.rho = 1.225            # 공기 밀도 (kg/m^3)
        self.drag_coeff = 2.5       # 항력 계수 (Drag) - 바람에 밀리는 힘
        self.lift_coeff = 0.5       # 양력 계수 (Lift) - 뜨는 힘
        self.wind_vel = cuda.to_device(np.array([0.0, 0.0, 8.0], dtype=np.float32)) # Z방향 강풍

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

        # compliance = 0.0       # 완전 딱딱함 (비신축성, 실크/면)
        # compliance = 0.00001   # 아주 약간 늘어남 (나일론)
        # compliance = 0.005     # 고무줄/스판덱스
        target_compliance = 0.0000001 # 거의 0에 가깝게 설정하여 기존 천 느낌 유지

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