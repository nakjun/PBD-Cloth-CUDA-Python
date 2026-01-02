import numpy as np
from numba import cuda
from PBD.module import predict_position_kernel, update_velocity_kernel
from PBD.module import solve_distance_constraint_colored_kernel
from PBD.module import compute_hash_kernel, find_cell_start_end_kernel, solve_self_collision_kernel
from PBD.coloring import compute_graph_coloring

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

        pos_host = np.zeros((num_particles, 3), dtype=np.float32)
        indices = []
        constraints = []

        # [SCENE SETUP] "The Squeezed Curtain"
        # 윗부분을 30%로 압축하여 강제로 주름을 만듭니다.
        compression_ratio = 0.5  # 1.0이면 평평함, 0.3이면 매우 쭈글쭈글함

        for y in range(height):
            for x in range(width):
                idx = y * width + x

                # 아코디언/커튼 효과: x축은 압축, z축은 사인파, 아래쪽이 펼쳐지게 y로 lerp할 수도 있음
                center_x = (width - 1) * spacing / 2.0
                original_x = x * spacing

                # 주름진 초기 상태: sine wave를 z축에 덧입힘
                freq = 1.5  # 주름의 빈도
                amp = spacing * 2.0 # 주름의 깊이

                z_offset = np.sin(x * freq) * amp

                # X축을 compression_ratio로 완전히 압축 (더 복잡하게 하려면 y에 따라 lerp 가능)
                pos_host[idx] = [x * spacing * compression_ratio, -y * spacing + (height * spacing), z_offset]

                # Structural Constraints 생성 (동일)
                if x < width - 1:
                    constraints.append([idx, idx + 1])
                if y < height - 1:
                    constraints.append([idx, idx + width])

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
        for x in range(width):
            mass_inv[x] = 0.0 
        self.d_mass_inv = cuda.to_device(mass_inv)

        # 제약 조건 GPU 할당
        constraints = np.array(constraints, dtype=np.int32)
        self.d_constraints = cuda.to_device(constraints)

        # Rest Length 계산
        rest_lengths = np.linalg.norm(pos_host[constraints[:, 0]] - pos_host[constraints[:, 1]], axis=1)
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
                self.num_particles, self.thickness
            )

            # 4. Update Velocity (기존)
            update_velocity_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos, self.d_vel, self.d_pos_pred, dt_sub, self.num_particles
            )

    def get_positions(self):
        return self.d_pos.copy_to_host()