import numpy as np
from numba import cuda
from PBD.module import predict_position_kernel, update_velocity_kernel
from PBD.module import solve_distance_constraint_colored_kernel
from PBD.module import compute_hash_kernel, find_cell_start_end_kernel, solve_self_collision_kernel
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

    def step(self):

        dt_sub = self.dt / self.substeps

        for _ in range(self.substeps):

            # 1. Predict (기존)
            predict_position_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos, self.d_vel, self.d_pos_pred, self.d_mass_inv, dt_sub, -9.8, self.num_particles
            )


            # 2. Graph Coloring Constraints (기존)
            # Color 0 실행 -> 동기화 -> Color 1 실행 -> 동기화 ...
            # k_stiffness = 1.0 (이제 1.0을 넣어도 폭발하지 않음!)
            for d_batch in self.d_color_batches:
                threads = 256
                blocks = (d_batch.shape[0] + threads - 1) // threads

                solve_distance_constraint_colored_kernel[blocks, threads](
                    self.d_pos_pred, self.d_mass_inv, self.d_constraints, self.d_rest_lengths,
                    d_batch, dt_sub, 1.0
                )
                # CUDA 커널 런칭은 기본적으로 비동기지만, 스트림이 같으면 순차 실행됨.
                # 필요하다면 cuda.synchronize()를 넣어도 됨.

            self.d_penetration[:] = 0.0 # 초기화
            

            # [NEW] 3. Self-Collision (Spatial Hashing)
            # 3-1. Reset Cell Info
            self.d_cell_start[:] = -1 # -1로 초기화 (비어있음)
            self.d_cell_end[:] = -1

            # 3-2. Compute Hash
            compute_hash_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos_pred, self.d_particle_hashes, self.d_particle_indices, self.num_particles
            )

            # 3-3. Sort (CPU로 가져와서 정렬 - Prototype용 타협)
            # *주의*: 고성능을 위해선 Thrust나 CuPy의 argsort를 써야 함.
            # 지금은 로직 검증용이라 numpy로 함.
            hashes = self.d_particle_hashes.copy_to_host()
            indices = self.d_particle_indices.copy_to_host()

            sort_order = np.argsort(hashes)
            sorted_hashes = hashes[sort_order]
            sorted_indices = indices[sort_order]

            # 다시 GPU로 업로드
            self.d_particle_hashes = cuda.to_device(sorted_hashes)
            self.d_particle_indices = cuda.to_device(sorted_indices)

            # 3-4. Find Cell Bounds
            find_cell_start_end_kernel[self.blocks_particles, self.threads_per_block](
                self.d_particle_hashes, self.d_cell_start, self.d_cell_end, self.num_particles
            )

            # 3-5. Solve Self Collision
            solve_self_collision_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos_pred, self.d_mass_inv, 
                self.d_cell_start, self.d_cell_end, 
                self.d_particle_indices, self.d_particle_hashes, 
                self.num_particles, self.thickness, self.d_penetration
            )

            # 4. Update Velocity (기존)
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