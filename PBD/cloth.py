import torch
import numpy as np
from numba import cuda
from PBD.module import predict_position_kernel, update_velocity_kernel
from PBD.module import solve_distance_constraint_colored_kernel
from PBD.module import (find_cell_start_end_kernel, 
                        solve_self_collision_friction_kernel, apply_aerodynamics_kernel,
                        solve_environment_collision_kernel, clear_counter_kernel, compute_curvature_kernel, compute_hash_kernel_v2,
                        compute_update_mask_kernel, compute_curvature_selective_kernel, count_updates_kernel,
                        compute_curvature_tiled_kernel, TILE_SIZE,
                        fused_curvature_hash_kernel, fused_curvature_hash_tiled_kernel,
                        fused_temporal_curvature_hash_kernel,
                        solve_distance_constraint_jacobi_kernel, apply_jacobi_correction_kernel)
from PBD.coloring import compute_graph_coloring
import math
import time

# 해시 테이블 상수
HASH_TABLE_SIZE = 1000003  # 해시 테이블 크기 (충분히 크게)
DEFAULT_CELL_SIZE = 0.1   # 기본 격자 크기

def compute_optimal_cell_size(spacing, thickness):
    """
    [Phase 2 최적화] 해상도에 따른 최적 셀 크기 계산
    
    셀 크기가 너무 작으면: 많은 셀 검색 필요 → 느림
    셀 크기가 너무 크면: 셀당 파티클 수 증가 → 느림
    
    최적: thickness * 2 ~ spacing * 2 사이
    """
    # 충돌 검출 거리의 2배 정도가 적당
    collision_distance = thickness * 2.0
    
    # spacing과 collision_distance 중 큰 값의 1.5배
    optimal = max(collision_distance, spacing) * 1.5
    
    # 최소/최대 제한
    return max(0.05, min(optimal, 0.5))

class ClothSimulator:
    def __init__(self, width, height, physical_width=10.0, dt=0.01, substeps=10, 
                 pinned_vertices=None, pin_mode='top_edge'):
        """
        [Resolution Independent Setup]
        width, height: 파티클 격자의 해상도 (개수)
        physical_width: 천의 실제 물리적 가로 길이 (미터)
        pinned_vertices: 고정할 vertex 인덱스 리스트 (직접 지정)
        pin_mode: 사전 정의된 고정 모드
            - 'none': 고정 없음 (기본값, 떨어지는 시나리오)
            - 'top_edge': 상단 가장자리 전체 고정 (커튼/깃발)
            - 'top_corners': 상단 양쪽 모서리만 고정
            - 'four_corners': 네 모서리 고정
            - 'top_row': 맨 위 한 줄 고정
            - 'custom': pinned_vertices로 직접 지정
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
        
        # [NEW] 고정 vertex 설정
        self.pin_mode = pin_mode
        self.pinned_indices = self._compute_pinned_indices(width, height, pin_mode, pinned_vertices)

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
        
        # [NEW] 고정 vertex는 mass_inv = 0 (움직이지 않음)
        for idx in self.pinned_indices:
            mass_inv[idx] = 0.0
        
        self.d_mass_inv = cuda.to_device(mass_inv)
        
        if len(self.pinned_indices) > 0:
            print(f"   -> Pinned Vertices: {len(self.pinned_indices)} particles (mode='{self.pin_mode}')")

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
        
        # [Phase 2 최적화] 동적 셀 크기 계산
        self.cell_size = compute_optimal_cell_size(spacing, self.thickness)
        print(f"   -> [OPT] Dynamic Cell Size: {self.cell_size:.4f} (spacing={spacing:.4f}, thickness={self.thickness:.4f})")
        
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
        # [개선] 해상도 독립성을 위한 정규화 파라미터
        # 곡률 임계값은 정규화된 값으로, 해상도에 무관하게 동일한 값 사용 가능
        self.spacing_sq = spacing * spacing  # h² for normalization
        self.curvature_threshold = 0.15      # 정규화된 임계값 (이전 0.002에서 조정)
        self.use_curvature_culling = True    # 곡률 기반 컬링 활성화 (기본값)
        
        # [GPU 최적화] Shared Memory Tiling 사용 여부
        # 고해상도(512x512 이상)에서 자동 활성화하여 Global Memory 접근 최소화
        self.use_tiled_curvature = (width >= 256 and height >= 256)
        
        # =============================================================================
        # [Phase 1 최적화] 융합 커널 사용 여부
        # =============================================================================
        self.use_fused_kernel = True  # Curvature + Hash 융합 커널 사용
        
        # =============================================================================
        # [Phase 1 최적화] Adaptive Curvature Threshold
        # =============================================================================
        self.adaptive_threshold_enabled = True
        self.base_curvature_threshold = 0.15  # 기본 임계값
        self.min_curvature_threshold = 0.08   # 최소 임계값 (격렬한 변형 시)
        self.max_curvature_threshold = 0.30   # 최대 임계값 (안정 시)
        self.last_avg_velocity = 0.0          # 이전 프레임 평균 속도
        self.last_max_penetration = 0.0       # 이전 프레임 최대 침투
        
        # =============================================================================
        # [Phase 3 최적화] Adaptive Substeps
        # =============================================================================
        self.adaptive_substeps_enabled = True
        self.base_substeps = substeps
        self.min_substeps = max(5, substeps // 2)   # 최소 substep (안정 시)
        self.max_substeps = substeps + 5            # 최대 substep (불안정 시)
        self.current_substeps = substeps            # 현재 사용 중인 substep 수
        
        # =============================================================================
        # [Phase 4 최적화] Jacobi Distance Constraint (선택적)
        # =============================================================================
        self.use_jacobi_constraint = False  # 기본값: Graph Coloring 방식 사용
        self.jacobi_relaxation = 0.8        # Under-relaxation 계수
        self.jacobi_iterations = 2          # Jacobi iteration 수
        
        # Jacobi용 버퍼
        self.d_pos_delta = cuda.device_array((self.num_particles, 3), dtype=np.float32)
        self.d_delta_count = cuda.device_array(self.num_particles, dtype=np.int32)

        # =============================================================================
        # [NEW] 시공간 코히어런스 (Spatio-Temporal Coherence) 버퍼
        # =============================================================================
        # 캐시된 곡률 값
        self.d_curvature_cache = cuda.device_array(self.num_particles, dtype=np.float32)
        # 캐시 시점의 위치
        self.d_pos_cache = cuda.to_device(pos_host.copy())
        # 캐시된 이후 경과한 substep 수
        self.d_cache_age = cuda.device_array(self.num_particles, dtype=np.int32)
        # 이번 substep에서 갱신 필요 여부
        self.d_update_mask = cuda.device_array(self.num_particles, dtype=np.bool_)
        
        # 초기화: 첫 프레임은 모두 계산하도록 강제
        self.d_cache_age[:] = 999
        self.d_curvature_cache[:] = 0.0
        
        # 시공간 코히어런스 파라미터
        self.motion_threshold = self.spacing * 0.05  # 권장: spacing * 0.03 ~ 0.07
        self.max_cache_age = 5  # 권장: 4 ~ 6 substeps
        
        # 통계/디버깅용 카운터
        self.d_update_counter = cuda.to_device(np.array([0], dtype=np.int32))
        self.d_active_pair_count = cuda.to_device(np.array([0], dtype=np.int32))
        self.last_update_ratio = 1.0  # 마지막 프레임의 갱신 비율
        
        # 시공간 코히어런스 활성화 플래그
        self.use_temporal_coherence = True
        
        print(f"   -> [NEW] Temporal Coherence: motion_threshold={self.motion_threshold:.6f}, max_cache_age={self.max_cache_age}")
        print(f"   -> [OPT] Fused Kernel: {self.use_fused_kernel}, Adaptive Threshold: {self.adaptive_threshold_enabled}")
        print(f"   -> [OPT] Adaptive Substeps: {self.adaptive_substeps_enabled} (range: {self.min_substeps}-{self.max_substeps})")

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
        """
        [최적화] 단순화된 정렬 - GPU 동기화 최소화
        
        이전 방식의 문제점:
        1. valid_mask.sum().item() → GPU→CPU 동기화 (블로킹)
        2. 마스킹 + 인덱싱 + cat → 추가 메모리 할당
        
        새로운 방식:
        - 전체 배열을 그대로 정렬 (hash=-1인 파티클은 앞으로 정렬됨)
        - GPU 동기화 최소화
        - 메모리 할당 최소화
        """
        # 1. [Sync Numba] 시작 전 대기
        cuda.synchronize()

        # 타이머 시작
        t_start = time.perf_counter() 

        # --- PyTorch 영역 ---
        hashes_torch = torch.as_tensor(self.d_particle_hashes, device='cuda')
        indices_torch = torch.as_tensor(self.d_particle_indices, device='cuda')
        
        # [최적화] 단순 정렬 - hash=-1은 자연스럽게 앞으로 정렬됨
        # argsort는 안정 정렬이므로 동일 해시값의 상대 순서 유지
        sorted_order = torch.argsort(hashes_torch, descending=False, stable=True)
        hashes_sorted = hashes_torch[sorted_order].contiguous()
        indices_sorted = indices_torch[sorted_order].contiguous()
        
        # 컬링 비율 계산 (hash != -1인 파티클 개수)
        num_valid = (hashes_torch != -1).sum().item()
        
        # [Sync PyTorch] PyTorch 작업 완료 대기
        torch.cuda.synchronize()
        # --- PyTorch 영역 끝 ---

        # 3. PyTorch -> Numba (데이터 덮어쓰기)
        sorted_hashes_cuda = cuda.as_cuda_array(hashes_sorted)
        sorted_indices_cuda = cuda.as_cuda_array(indices_sorted)
        
        self.d_particle_hashes.copy_to_device(sorted_hashes_cuda)
        self.d_particle_indices.copy_to_device(sorted_indices_cuda)

        cuda.synchronize()

        t_end = time.perf_counter()
        self.last_sort_time = t_end - t_start
        
        # 디버깅용: 컬링 비율 저장
        self.last_culling_ratio = 1.0 - (num_valid / self.num_particles) if self.num_particles > 0 else 0.0

    def _update_adaptive_parameters(self):
        """
        [Phase 1 & 3 최적화] 시뮬레이션 상태에 따라 파라미터 동적 조절
        """
        # =================================================================
        # Adaptive Curvature Threshold
        # =================================================================
        if self.adaptive_threshold_enabled:
            # 속도 기반 조절: 빠른 움직임 → 낮은 임계값 (더 정확)
            velocity_factor = min(1.0, self.last_avg_velocity / 5.0)  # 0~1 정규화
            
            # 침투 기반 조절: 높은 침투 → 낮은 임계값 (더 정확)
            penetration_factor = min(1.0, self.last_max_penetration / 0.01)  # 0~1 정규화
            
            # 종합 요소 (높을수록 불안정)
            instability = max(velocity_factor, penetration_factor)
            
            # 임계값 계산: 불안정할수록 낮은 임계값
            threshold_range = self.max_curvature_threshold - self.min_curvature_threshold
            self.curvature_threshold = self.max_curvature_threshold - (instability * threshold_range)
        
        # =================================================================
        # Adaptive Substeps
        # =================================================================
        if self.adaptive_substeps_enabled:
            # 속도와 침투 기반으로 substep 수 결정
            if self.last_avg_velocity < 1.0 and self.last_max_penetration < 0.002:
                # 안정 상태: 적은 substep
                self.current_substeps = self.min_substeps
            elif self.last_avg_velocity > 8.0 or self.last_max_penetration > 0.008:
                # 불안정 상태: 많은 substep
                self.current_substeps = self.max_substeps
            else:
                # 중간 상태: 기본 substep
                self.current_substeps = self.base_substeps
    
    def step(self):
        target_compliance = 0.0       # 완전 딱딱함 (비신축성, 실크/면)
        self.frame_count += 1
        
        # =================================================================
        # [Phase 1 & 3 최적화] Adaptive 파라미터 업데이트
        # =================================================================
        self._update_adaptive_parameters()
        
        # Adaptive Substeps 적용
        current_substeps = self.current_substeps if self.adaptive_substeps_enabled else self.substeps
        dt_sub = self.dt / current_substeps

        for _ in range(current_substeps):
                
            # ------------------------------------------------------------------
            # [Stage 0] External Forces (Aerodynamics)
            # ------------------------------------------------------------------
            blocks_faces = (self.num_faces + self.threads_per_block - 1) // self.threads_per_block
            
            apply_aerodynamics_kernel[blocks_faces, self.threads_per_block](
                self.d_pos, self.d_vel, self.d_faces, self.wind_vel,
                self.rho, self.drag_coeff, self.lift_coeff, dt_sub, self.num_faces
            )

            # ------------------------------------------------------------------
            # [Stage 1] Prediction & Integration
            # ------------------------------------------------------------------
            predict_position_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos, self.d_vel, self.d_pos_pred, self.d_mass_inv, 
                dt_sub, -9.8, self.num_particles
            )

            # ------------------------------------------------------------------
            # [Stage 1.5] Environment Collision (Sphere SDF)
            # ------------------------------------------------------------------
            solve_environment_collision_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos_pred, self.d_pos, self.d_mass_inv,
                self.sphere_params, self.sphere_friction,
                self.floor_height, self.floor_friction,
                dt_sub, self.num_particles, self.collision_margin
            )
            
            # ------------------------------------------------------------------
            # [Stage 2] Distance Constraints (XPBD)
            # ------------------------------------------------------------------
            if self.use_jacobi_constraint:
                # [Phase 4 최적화] Jacobi 방식 (단일 커널)
                blocks_constraints = (self.num_constraints + self.threads_per_block - 1) // self.threads_per_block
                
                for _ in range(self.jacobi_iterations):
                    # 버퍼 초기화
                    self.d_pos_delta[:] = 0.0
                    self.d_delta_count[:] = 0
                    
                    # 모든 제약 병렬 처리
                    solve_distance_constraint_jacobi_kernel[blocks_constraints, self.threads_per_block](
                        self.d_pos_pred, self.d_pos_delta, self.d_delta_count,
                        self.d_mass_inv, self.d_constraints, self.d_rest_lengths,
                        dt_sub, target_compliance, self.num_constraints
                    )
                    
                    # 보정값 적용 (평균화 + under-relaxation)
                    apply_jacobi_correction_kernel[self.blocks_particles, self.threads_per_block](
                        self.d_pos_pred, self.d_pos_delta, self.d_delta_count,
                        self.jacobi_relaxation, self.num_particles
                    )
            else:
                # 기존 방식: Graph Coloring
                for d_batch in self.d_color_batches:
                    blocks_batch = (d_batch.shape[0] + self.threads_per_block - 1) // self.threads_per_block
                    solve_distance_constraint_colored_kernel[blocks_batch, self.threads_per_block](
                        self.d_pos_pred, self.d_mass_inv, self.d_constraints, self.d_rest_lengths,
                        d_batch, dt_sub, target_compliance
                    )

            # ------------------------------------------------------------------
            # [Stage 3] Self-Collision (Optimized Pipeline)
            # ------------------------------------------------------------------
            # 3-0. Reset Grid
            self.d_cell_start[:] = -1 
            self.d_cell_end[:] = -1
            self.d_penetration[:] = 0.0

            # =================================================================
            # [최적화] 단계적 컬링 파이프라인 - 융합 커널 우선 사용
            # =================================================================
            if self.use_curvature_culling and self.use_temporal_coherence:
                # [최적화 경로 1] Temporal Coherence + Curvature + Hash 융합 커널
                # full_optimization에서 3개 커널을 1개로 통합하여 오버헤드 제거
                fused_temporal_curvature_hash_kernel[self.blocks_per_grid_2d, self.threads_per_block_2d](
                    self.d_pos, self.d_pos_pred,
                    self.d_pos_cache, self.d_cache_age, self.d_update_mask,
                    self.d_curvature, self.d_curvature_cache,
                    self.d_particle_hashes, self.d_particle_indices,
                    self.num_x, self.num_y, self.spacing_sq,
                    self.curvature_threshold, self.motion_threshold, self.max_cache_age,
                    self.cell_size, HASH_TABLE_SIZE
                )
            elif self.use_fused_kernel and not self.use_temporal_coherence:
                # [최적화 경로 2] Curvature + Hash 융합 커널 (Temporal Coherence 없음)
                if self.use_tiled_curvature:
                    # Tiled 융합 커널 (Shared Memory 활용)
                    tiled_blocks_x = int(np.ceil(self.num_x / TILE_SIZE))
                    tiled_blocks_y = int(np.ceil(self.num_y / TILE_SIZE))
                    fused_curvature_hash_tiled_kernel[(tiled_blocks_x, tiled_blocks_y), (TILE_SIZE, TILE_SIZE)](
                        self.d_pos, self.d_pos_pred,
                        self.d_curvature, self.d_particle_hashes, self.d_particle_indices,
                        self.num_x, self.num_y, self.spacing_sq,
                        self.curvature_threshold, self.cell_size, HASH_TABLE_SIZE
                    )
                else:
                    # 기본 융합 커널
                    fused_curvature_hash_kernel[self.blocks_per_grid_2d, self.threads_per_block_2d](
                        self.d_pos, self.d_pos_pred,
                        self.d_curvature, self.d_particle_hashes, self.d_particle_indices,
                        self.num_x, self.num_y, self.spacing_sq,
                        self.curvature_threshold, self.cell_size, HASH_TABLE_SIZE
                    )
            else:
                # [기존 경로] 별도 커널 사용 (Temporal Coherence만 사용하거나 컬링 없음)
                threads_1d = 256
                blocks_1d = int(math.ceil(self.num_particles / threads_1d))
                
                if self.use_temporal_coherence:
                    # 시공간 코히어런스 적용 (이 경로는 위에서 처리되므로 여기 도달하지 않음)
                    compute_update_mask_kernel[self.blocks_per_grid_2d, self.threads_per_block_2d](
                        self.d_pos, self.d_pos_cache, self.d_cache_age, self.d_update_mask,
                        self.motion_threshold, self.max_cache_age, self.num_x, self.num_y
                    )
                    
                    compute_curvature_selective_kernel[self.blocks_per_grid_2d, self.threads_per_block_2d](
                        self.d_pos, self.d_curvature, self.d_curvature_cache,
                        self.d_pos_cache, self.d_cache_age, self.d_update_mask,
                        self.num_x, self.num_y, self.spacing_sq
                    )
                else:
                    # 기존 방식
                    if self.use_tiled_curvature:
                        tiled_blocks_x = int(np.ceil(self.num_x / TILE_SIZE))
                        tiled_blocks_y = int(np.ceil(self.num_y / TILE_SIZE))
                        compute_curvature_tiled_kernel[(tiled_blocks_x, tiled_blocks_y), (TILE_SIZE, TILE_SIZE)](
                            self.d_pos, self.d_curvature, self.num_x, self.num_y, self.spacing_sq
                        )
                    else:
                        compute_curvature_kernel[self.blocks_per_grid_2d, self.threads_per_block_2d](
                            self.d_pos, self.d_curvature, self.num_x, self.num_y, self.spacing_sq
                        )
                
                # 해시 계산 (별도 커널)
                compute_hash_kernel_v2[blocks_1d, threads_1d](
                    self.d_pos_pred, self.d_particle_hashes, self.d_particle_indices,
                    self.d_cell_start, self.d_cell_end, self.d_curvature,
                    self.curvature_threshold, self.num_particles, CELL_SIZE, HASH_TABLE_SIZE
                )
            # =================================================================

            # 3-3. Sort Particles (PyTorch Radix Sort - Zero Copy)
            self._sort_particles_torch()

            # 3-4. Find Cell Bounds
            find_cell_start_end_kernel[self.blocks_particles, self.threads_per_block](
                self.d_particle_hashes, self.d_cell_start, self.d_cell_end, self.num_particles
            )

            # 3-5. Solve Collision with Friction (다단계 Culling 적용)
            # Curvature threshold: use_curvature_culling이 True면 실제 임계값, 아니면 0 (모든 파티클 통과)
            curv_thresh = self.curvature_threshold if self.use_curvature_culling else 0.0
            
            # Active pair 카운터 초기화
            self.d_active_pair_count[0] = 0
            
            solve_self_collision_friction_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos_pred,        # Candidate Position
                self.d_pos,             # Previous Position (for relative velocity)
                self.d_mass_inv, 
                self.d_cell_start, self.d_cell_end, 
                self.d_particle_indices, self.d_particle_hashes, 
                self.num_particles, self.thickness, self.d_penetration,
                dt_sub,
                self.d_visibility, self.frame_count,
                self.d_debug_skip_count,
                self.d_curvature, curv_thresh,
                self.d_active_pair_count
            )

            # ------------------------------------------------------------------
            # [Stage 4] Velocity Update
            # ------------------------------------------------------------------
            # 위치 확정 및 속도 갱신 (Damping 포함)
            update_velocity_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos, self.d_vel, self.d_pos_pred, dt_sub, self.num_particles
            )

        # =================================================================
        # [Phase 1 & 3 최적화] 다음 프레임을 위한 통계 수집
        # =================================================================
        if self.adaptive_threshold_enabled or self.adaptive_substeps_enabled:
            # 속도 통계 (GPU에서 직접 계산하면 더 빠르지만, 간단히 CPU에서 계산)
            # 매 프레임 전체 복사는 비효율적이므로, 샘플링 또는 주기적 업데이트 권장
            if self.frame_count % 5 == 0:  # 5프레임마다 업데이트
                vel_host = self.d_vel.copy_to_host()
                vel_magnitudes = np.linalg.norm(vel_host, axis=1)
                self.last_avg_velocity = np.mean(vel_magnitudes)
                
                penetration_host = self.d_penetration.copy_to_host()
                self.last_max_penetration = np.max(penetration_host)

    def get_positions(self):
        return self.d_pos.copy_to_host()

    def get_penetration_depth(self):
        return self.d_penetration.copy_to_host()

    def get_velocities(self):
        return self.d_vel.copy_to_host()

    def get_update_ratio(self):
        """
        [시공간 코히어런스 통계] 마지막 substep에서 실제로 갱신된 파티클의 비율을 반환
        Returns: float (0.0 ~ 1.0)
        """
        if not self.use_temporal_coherence:
            return 1.0
        
        # 카운터 리셋
        self.d_update_counter[:] = 0
        
        # 갱신 수 카운트
        threads_1d = 256
        blocks_1d = int(math.ceil(self.num_particles / threads_1d))
        count_updates_kernel[blocks_1d, threads_1d](
            self.d_update_mask,
            self.d_update_counter,
            self.num_particles
        )
        
        update_count = self.d_update_counter.copy_to_host()[0]
        self.last_update_ratio = update_count / self.num_particles
        return self.last_update_ratio
    
    def get_cache_hit_rate(self):
        """
        [시공간 코히어런스 통계] 캐시 히트율 (1 - update_ratio)
        Returns: float (0.0 ~ 1.0)
        """
        return 1.0 - self.get_update_ratio()
    
    # =========================================================================
    # [NEW] Pinned Vertex (고정 파티클) 관련 메서드
    # =========================================================================
    def _compute_pinned_indices(self, width, height, pin_mode, pinned_vertices):
        """
        pin_mode에 따라 고정할 vertex 인덱스 리스트 계산
        
        그리드 레이아웃:
        - idx = y * width + x
        - y=0: 맨 아래 (바닥 근처)
        - y=height-1: 맨 위
        - x=0: 왼쪽, x=width-1: 오른쪽
        """
        if pin_mode == 'none':
            return []
        
        if pin_mode == 'custom' and pinned_vertices is not None:
            return list(pinned_vertices)
        
        if pin_mode == 'top_edge':
            # 상단 가장자리 전체 (y = height - 1)
            return [((height - 1) * width + x) for x in range(width)]
        
        if pin_mode == 'top_corners':
            # 상단 양쪽 모서리만
            top_left = (height - 1) * width + 0
            top_right = (height - 1) * width + (width - 1)
            return [top_left, top_right]
        
        if pin_mode == 'four_corners':
            # 네 모서리
            top_left = (height - 1) * width + 0
            top_right = (height - 1) * width + (width - 1)
            bottom_left = 0
            bottom_right = width - 1
            return [top_left, top_right, bottom_left, bottom_right]
        
        if pin_mode == 'top_row':
            # 맨 위 한 줄 (= top_edge와 동일)
            return [((height - 1) * width + x) for x in range(width)]
        
        # 알 수 없는 모드
        print(f"   [Warning] Unknown pin_mode '{pin_mode}', no vertices pinned.")
        return []
    
    def pin_vertex(self, idx):
        """
        런타임에 특정 vertex를 고정
        
        Args:
            idx: 고정할 vertex 인덱스 (int) 또는 인덱스 리스트
        """
        if isinstance(idx, (list, tuple, np.ndarray)):
            indices = list(idx)
        else:
            indices = [idx]
        
        # GPU에서 mass_inv 가져오기
        mass_inv = self.d_mass_inv.copy_to_host()
        
        for i in indices:
            if 0 <= i < self.num_particles:
                mass_inv[i] = 0.0
                if i not in self.pinned_indices:
                    self.pinned_indices.append(i)
        
        # GPU로 다시 업로드
        self.d_mass_inv = cuda.to_device(mass_inv)
    
    def unpin_vertex(self, idx):
        """
        런타임에 특정 vertex 고정 해제
        
        Args:
            idx: 고정 해제할 vertex 인덱스 (int) 또는 인덱스 리스트
        """
        if isinstance(idx, (list, tuple, np.ndarray)):
            indices = list(idx)
        else:
            indices = [idx]
        
        # GPU에서 mass_inv 가져오기
        mass_inv = self.d_mass_inv.copy_to_host()
        
        # 원래 mass 계산
        ref_spacing = 0.1
        particle_mass = (self.spacing / ref_spacing) ** 2
        original_mass_inv = 1.0 / particle_mass
        
        for i in indices:
            if 0 <= i < self.num_particles:
                mass_inv[i] = original_mass_inv
                if i in self.pinned_indices:
                    self.pinned_indices.remove(i)
        
        # GPU로 다시 업로드
        self.d_mass_inv = cuda.to_device(mass_inv)
    
    def unpin_all(self):
        """모든 고정 해제"""
        self.unpin_vertex(self.pinned_indices.copy())
    
    def get_pinned_indices(self):
        """현재 고정된 vertex 인덱스 리스트 반환"""
        return self.pinned_indices.copy()
    
    def is_pinned(self, idx):
        """특정 vertex가 고정되어 있는지 확인"""
        return idx in self.pinned_indices
    
    def pin_by_position(self, condition_fn):
        """
        위치 조건에 따라 vertex 고정
        
        Args:
            condition_fn: (x, y, z) -> bool 함수
                          True를 반환하면 해당 vertex 고정
        
        Example:
            # y > 5.0인 모든 vertex 고정
            sim.pin_by_position(lambda x, y, z: y > 5.0)
        """
        pos = self.get_positions()
        indices_to_pin = []
        
        for i in range(self.num_particles):
            x, y, z = pos[i]
            if condition_fn(x, y, z):
                indices_to_pin.append(i)
        
        if indices_to_pin:
            self.pin_vertex(indices_to_pin)
            print(f"   -> Pinned {len(indices_to_pin)} vertices by position condition")
    
    def set_temporal_coherence(self, enabled, motion_threshold=None, max_cache_age=None):
        """
        시공간 코히어런스 설정 변경
        """
        self.use_temporal_coherence = enabled
        if motion_threshold is not None:
            self.motion_threshold = motion_threshold
        if max_cache_age is not None:
            self.max_cache_age = max_cache_age
        
        if enabled:
            # 활성화 시 캐시 강제 리셋
            self.d_cache_age[:] = 999
            
        print(f"[Temporal Coherence] enabled={enabled}, motion_threshold={self.motion_threshold:.6f}, max_cache_age={self.max_cache_age}")

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