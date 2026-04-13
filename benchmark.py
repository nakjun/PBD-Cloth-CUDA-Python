"""
=============================================================================
Cloth Simulation Comprehensive Benchmark System
=============================================================================
벤치마크 실험 설정:
1. Culling 알고리즘 Ablation (Self-Collision Only, Spatial Hashing, Temporal Coherence, compute_hash_kernel_v2)
2. Cloth Model Size Ablation (128x128, 256x256, 512x512, 1024x1024, 2048x2048)
3. 2000 프레임 테스트, 소수점 셋째자리까지 저장
4. 통합 CSV 파일로 결과 저장
5. 결과 시각화 차트 생성
"""

import sys
import os
import numpy as np
import math
import time
import csv
from datetime import datetime
from tqdm import tqdm
from numba import cuda
import gc

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# ============================================================
# 벤치마크 설정
# ============================================================

# 테스트할 cloth 사이즈
CLOTH_SIZES = [128, 256, 512, 1024, 2048]

# Culling 알고리즘 ablation 설정
# 각 설정은 (이름, use_spatial_hashing, use_temporal_coherence, use_curvature_culling) 튜플
# 참고: Spatial Hashing은 3D 공간에서 접히는 천의 충돌을 감지하기 위해 필수입니다.
#       Baseline은 Spatial Hashing만 사용하고 추가 최적화는 적용하지 않습니다.
# [하이브리드 접근법] 3개 설정으로 핵심 기법의 효과와 전체 최적화 효과를 측정
CULLING_CONFIGS = [
    ("baseline_spatial_hashing", True, False, False),      # Baseline: Spatial Hashing만 사용
    ("curvature_culling", True, False, True),              # + Curvature Culling (핵심 기법)
    ("full_optimization", True, True, True),               # 모든 최적화 적용 (Temporal + Curvature)
]

# 공통 설정
PHYSICAL_WIDTH = 12.0
DT = 0.01
SUBSTEPS = 15
TOTAL_FRAMES = 2000  # 2000 프레임 테스트
NUM_TRIALS = 5       # 반복 실험 횟수 (mean±std 계산용)

# 결과 저장 경로
RESULTS_DIR = os.path.join(current_dir, "benchmark_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class BenchmarkClothSimulator:
    """
    벤치마크용 ClothSimulator - Culling 알고리즘 on/off 지원
    """
    def __init__(self, width, height, physical_width=10.0, dt=0.01, substeps=10,
                 use_spatial_hashing=True, use_temporal_coherence=True, use_curvature_culling=True):
        """
        벤치마크용 시뮬레이터 초기화
        
        Args:
            use_spatial_hashing: Spatial Hashing 사용 여부
            use_temporal_coherence: Temporal Coherence 사용 여부  
            use_curvature_culling: compute_hash_kernel_v2 (곡률 기반 컬링) 사용 여부
        """
        import torch
        from PBD.module import predict_position_kernel, update_velocity_kernel
        from PBD.module import solve_distance_constraint_colored_kernel
        from PBD.module import (find_cell_start_end_kernel, 
                                solve_self_collision_friction_kernel, apply_aerodynamics_kernel,
                                solve_environment_collision_kernel, clear_counter_kernel, 
                                compute_curvature_kernel, compute_hash_kernel_v2,
                                compute_update_mask_kernel, compute_curvature_selective_kernel, 
                                count_updates_kernel, solve_self_collision_bruteforce_kernel,
                                compute_curvature_tiled_kernel, TILE_SIZE,
                                fused_curvature_hash_kernel, fused_curvature_hash_tiled_kernel,
                                fused_temporal_curvature_hash_kernel)
        from PBD.coloring import compute_graph_coloring
        
        # 모듈 참조 저장
        self.torch = torch
        self.predict_position_kernel = predict_position_kernel
        self.update_velocity_kernel = update_velocity_kernel
        self.solve_distance_constraint_colored_kernel = solve_distance_constraint_colored_kernel
        self.find_cell_start_end_kernel = find_cell_start_end_kernel
        self.solve_self_collision_friction_kernel = solve_self_collision_friction_kernel
        self.solve_self_collision_bruteforce_kernel = solve_self_collision_bruteforce_kernel
        self.apply_aerodynamics_kernel = apply_aerodynamics_kernel
        self.solve_environment_collision_kernel = solve_environment_collision_kernel
        self.compute_curvature_kernel = compute_curvature_kernel
        self.compute_curvature_tiled_kernel = compute_curvature_tiled_kernel
        self.TILE_SIZE = TILE_SIZE
        self.compute_hash_kernel_v2 = compute_hash_kernel_v2
        self.compute_update_mask_kernel = compute_update_mask_kernel
        self.compute_curvature_selective_kernel = compute_curvature_selective_kernel
        # [Phase 1 최적화] 융합 커널
        self.fused_curvature_hash_kernel = fused_curvature_hash_kernel
        self.fused_curvature_hash_tiled_kernel = fused_curvature_hash_tiled_kernel
        self.fused_temporal_curvature_hash_kernel = fused_temporal_curvature_hash_kernel
        
        # Ablation 설정
        self.use_spatial_hashing = use_spatial_hashing
        self.use_temporal_coherence = use_temporal_coherence
        self.use_curvature_culling = use_curvature_culling
        
        self.dt = dt
        self.substeps = substeps
        
        # 해시 테이블 상수
        self.HASH_TABLE_SIZE = 1000003
        
        # Resolution 설정
        self.num_x = width
        self.num_y = height
        num_particles = width * height
        self.num_particles = num_particles
        
        spacing = physical_width / (width - 1)
        self.spacing = spacing
        
        # [Phase 2 최적화] 동적 셀 크기 계산
        thickness = spacing * 0.7
        collision_distance = thickness * 2.0
        self.CELL_SIZE = max(0.05, min(max(collision_distance, spacing) * 1.5, 0.5))
        physical_height = spacing * (height - 1)

        # Sphere 설정
        sphere_radius = physical_width * 0.3
        sphere_center_x = physical_width * 0.5 + sphere_radius
        sphere_center_y = sphere_radius + 0.3
        sphere_center_z = physical_height * 0.5
        sphere_center = np.array([sphere_center_x, sphere_center_y, sphere_center_z], dtype=np.float32)
        self.sphere_params = cuda.to_device(np.concatenate([sphere_center, [sphere_radius]]))

        self.floor_height = 0.0
        self.floor_friction = 0.9
        self.sphere_friction = 0.02

        # 파티클 초기 위치
        pos_host = np.zeros((num_particles, 3), dtype=np.float32)
        start_y = sphere_center_y + sphere_radius + (physical_width * 0.1)
        
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                pos_x = x * spacing
                pos_z = y * spacing
                pos_y = start_y + np.random.uniform(-spacing*0.1, spacing*0.1)
                pos_host[idx] = [pos_x, pos_y, pos_z]

        # 제약 조건 생성
        constraints = []
        rest_lengths_list = []

        for y in range(height):
            for x in range(width):
                idx = y * width + x
                if x < width - 1: 
                    constraints.append([idx, idx + 1])
                    rest_lengths_list.append(spacing)
                if y < height - 1: 
                    constraints.append([idx, idx + width])
                    rest_lengths_list.append(spacing)
                
                diag_dist = spacing * math.sqrt(2)
                if x < width - 1 and y < height - 1:
                    constraints.append([idx, idx + width + 1])
                    rest_lengths_list.append(diag_dist)
                    constraints.append([idx + 1, idx + width])
                    rest_lengths_list.append(diag_dist)

        self.num_constraints = len(constraints)

        # Graph Coloring
        color_batches_host = compute_graph_coloring(num_particles, constraints)
        self.d_color_batches = []
        for batch in color_batches_host:
            self.d_color_batches.append(cuda.to_device(batch))

        # GPU 데이터 할당
        self.d_pos = cuda.to_device(pos_host)
        self.d_pos_pred = cuda.to_device(pos_host)
        self.d_vel = cuda.to_device(np.zeros_like(pos_host))

        ref_spacing = 0.1
        particle_mass = (spacing / ref_spacing) ** 2 
        mass_inv = np.ones(num_particles, dtype=np.float32) * (1.0 / particle_mass)
        self.d_mass_inv = cuda.to_device(mass_inv)

        self.d_constraints = cuda.to_device(np.array(constraints, dtype=np.int32))
        self.d_rest_lengths = cuda.to_device(np.array(rest_lengths_list, dtype=np.float32))

        # Spatial Hashing 버퍼
        self.d_particle_hashes = cuda.device_array(self.num_particles, dtype=np.int32)
        self.d_particle_indices = cuda.device_array(self.num_particles, dtype=np.int32)
        self.d_cell_start = cuda.device_array(self.HASH_TABLE_SIZE, dtype=np.int32)
        self.d_cell_end = cuda.device_array(self.HASH_TABLE_SIZE, dtype=np.int32)
        
        self.thickness = spacing * 0.7
        self.collision_margin = self.thickness * 0.5
        self.d_penetration = cuda.device_array(self.num_particles, dtype=np.float32)

        # Faces 생성
        faces_list = []
        for y in range(height - 1):
            for x in range(width - 1):
                idx = y * width + x
                faces_list.append([idx, idx + 1, idx + width + 1])
                faces_list.append([idx, idx + width + 1, idx + width])
        
        faces_array = np.array(faces_list, dtype=np.int32)
        self.num_faces = len(faces_array)
        self.d_faces = cuda.to_device(faces_array)

        # View-Dependent Culling 버퍼
        self.frame_count = 0
        self.d_normals = cuda.device_array((self.num_particles, 3), dtype=np.float32)
        self.d_visibility = cuda.device_array(self.num_particles, dtype=np.float32)
        
        camera_pos_host = np.array([self.num_x * self.spacing * 0.5, 10.0, 
                                    self.num_y * self.spacing * 1.5], dtype=np.float32)
        self.d_camera_pos = cuda.to_device(camera_pos_host)

        # Aerodynamics
        self.rho = 1.225
        self.drag_coeff = 2.5
        self.lift_coeff = 0.5
        self.wind_vel = cuda.to_device(np.array([0.0, 0.0, 3.0], dtype=np.float32))

        # CUDA 설정
        self.threads_per_block = 256
        self.blocks_particles = (self.num_particles + self.threads_per_block - 1) // self.threads_per_block
        self.threads_per_block_2d = (16, 16)
        self.blocks_per_grid_x = int(np.ceil(self.num_x / 16))
        self.blocks_per_grid_y = int(np.ceil(self.num_y / 16))
        self.blocks_per_grid_2d = (self.blocks_per_grid_x, self.blocks_per_grid_y)

        self.d_debug_skip_count = cuda.to_device(np.array([0], dtype=np.int32))
        self.d_active_pair_count = cuda.to_device(np.array([0], dtype=np.int32))
        self.last_sort_time = 0.0

        # 곡률 기반 컬링
        self.d_curvature = cuda.device_array(self.num_particles, dtype=np.float32)
        # [개선] 해상도 독립성을 위한 정규화 파라미터
        self.spacing_sq = spacing * spacing  # h² for normalization
        self.curvature_threshold = 0.15 if use_curvature_culling else 0.0  # 정규화된 임계값
        
        # [GPU 최적화] Shared Memory Tiling 사용 여부
        # 고해상도(256x256 이상)에서 자동 활성화하여 Global Memory 접근 최소화
        self.use_tiled_curvature = (width >= 256 and height >= 256)
        
        # [Phase 1 최적화] 융합 커널 사용 (Temporal Coherence 미사용 시)
        self.use_fused_kernel = use_curvature_culling and not use_temporal_coherence

        # Temporal Coherence 버퍼
        self.d_curvature_cache = cuda.device_array(self.num_particles, dtype=np.float32)
        self.d_pos_cache = cuda.to_device(pos_host.copy())
        self.d_cache_age = cuda.device_array(self.num_particles, dtype=np.int32)
        self.d_update_mask = cuda.device_array(self.num_particles, dtype=np.bool_)
        
        self.d_cache_age[:] = 999
        self.d_curvature_cache[:] = 0.0
        
        self.motion_threshold = self.spacing * 0.05
        self.max_cache_age = 5
        self.d_update_counter = cuda.to_device(np.array([0], dtype=np.int32))
        self.last_update_ratio = 1.0

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
        cuda.synchronize()
        t_start = time.perf_counter()

        hashes_torch = self.torch.as_tensor(self.d_particle_hashes, device='cuda')
        indices_torch = self.torch.as_tensor(self.d_particle_indices, device='cuda')
        
        # [최적화] 단순 정렬 - hash=-1은 자연스럽게 앞으로 정렬됨
        # argsort는 안정 정렬이므로 동일 해시값의 상대 순서 유지
        sorted_order = self.torch.argsort(hashes_torch, descending=False, stable=True)
        hashes_sorted = hashes_torch[sorted_order].contiguous()
        indices_sorted = indices_torch[sorted_order].contiguous()
        
        self.torch.cuda.synchronize()

        sorted_hashes_cuda = cuda.as_cuda_array(hashes_sorted)
        sorted_indices_cuda = cuda.as_cuda_array(indices_sorted)
        
        self.d_particle_hashes.copy_to_device(sorted_hashes_cuda)
        self.d_particle_indices.copy_to_device(sorted_indices_cuda)

        cuda.synchronize()
        t_end = time.perf_counter()
        self.last_sort_time = t_end - t_start

    def _compute_hash_baseline(self):
        """Baseline: 단순 해시 계산 (곡률 컬링 없음)"""
        threads_1d = 256
        blocks_1d = int(math.ceil(self.num_particles / threads_1d))
        
        # 간단한 해시 커널 (곡률 임계값 0으로 설정하여 모든 파티클 포함)
        self.compute_hash_kernel_v2[blocks_1d, threads_1d](
            self.d_pos_pred,
            self.d_particle_hashes,
            self.d_particle_indices,
            self.d_cell_start,
            self.d_cell_end,
            self.d_curvature,
            0.0,  # threshold = 0 -> 모든 파티클 포함
            self.num_particles,
            self.CELL_SIZE,
            self.HASH_TABLE_SIZE
        )

    def step(self):
        target_compliance = 0.0
        self.frame_count += 1
        dt_sub = self.dt / self.substeps

        for _ in range(self.substeps):
            # Aerodynamics
            blocks_faces = (self.num_faces + self.threads_per_block - 1) // self.threads_per_block
            self.apply_aerodynamics_kernel[blocks_faces, self.threads_per_block](
                self.d_pos, self.d_vel, self.d_faces, self.wind_vel,
                self.rho, self.drag_coeff, self.lift_coeff, dt_sub, self.num_faces
            )

            # Prediction
            self.predict_position_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos, self.d_vel, self.d_pos_pred, self.d_mass_inv, 
                dt_sub, -9.8, self.num_particles
            )

            # Environment Collision
            self.solve_environment_collision_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos_pred, self.d_pos, self.d_mass_inv,
                self.sphere_params, self.sphere_friction,
                self.floor_height, self.floor_friction,
                dt_sub, self.num_particles, self.collision_margin
            )

            # Distance Constraints
            for d_batch in self.d_color_batches:
                blocks_batch = (d_batch.shape[0] + self.threads_per_block - 1) // self.threads_per_block
                self.solve_distance_constraint_colored_kernel[blocks_batch, self.threads_per_block](
                    self.d_pos_pred, self.d_mass_inv, self.d_constraints, self.d_rest_lengths,
                    d_batch, dt_sub, target_compliance
                )

            # Self-Collision (Ablation에 따라 분기)
            # 모든 설정에서 Spatial Hashing 사용 (3D 충돌 감지 필수)
            # 최적화 옵션만 on/off
            self._step_with_spatial_hashing(dt_sub)

            # Velocity Update
            self.update_velocity_kernel[self.blocks_particles, self.threads_per_block](
                self.d_pos, self.d_vel, self.d_pos_pred, dt_sub, self.num_particles
            )

    def _step_with_spatial_hashing(self, dt_sub):
        """
        Spatial Hashing을 사용하는 충돌 처리 (최적화됨)
        [Phase 1 최적화] 융합 커널 지원
        [Phase 2 최적화] 동적 셀 크기
        """
        self.d_cell_start[:] = -1 
        self.d_cell_end[:] = -1
        self.d_penetration[:] = 0.0

        threads_1d = 256
        blocks_1d = int(math.ceil(self.num_particles / threads_1d))

        threshold = 0.0
        if self.use_curvature_culling:
            threshold = self.curvature_threshold
        
        # =================================================================
        # [최적화] 단계적 컬링 파이프라인 - 융합 커널 우선 사용
        # =================================================================
        if self.use_curvature_culling and self.use_temporal_coherence:
            # [최적화 경로 1] Temporal Coherence + Curvature + Hash 융합 커널
            # full_optimization에서 3개 커널을 1개로 통합하여 오버헤드 제거
            self.fused_temporal_curvature_hash_kernel[self.blocks_per_grid_2d, self.threads_per_block_2d](
                self.d_pos, self.d_pos_pred,
                self.d_pos_cache, self.d_cache_age, self.d_update_mask,
                self.d_curvature, self.d_curvature_cache,
                self.d_particle_hashes, self.d_particle_indices,
                self.num_x, self.num_y, self.spacing_sq,
                threshold, self.motion_threshold, self.max_cache_age,
                self.CELL_SIZE, self.HASH_TABLE_SIZE
            )
        elif self.use_fused_kernel and self.use_curvature_culling:
            # [최적화 경로 2] Curvature + Hash 융합 커널 (Temporal Coherence 없음)
            if self.use_tiled_curvature:
                tiled_blocks_x = int(math.ceil(self.num_x / self.TILE_SIZE))
                tiled_blocks_y = int(math.ceil(self.num_y / self.TILE_SIZE))
                self.fused_curvature_hash_tiled_kernel[(tiled_blocks_x, tiled_blocks_y), (self.TILE_SIZE, self.TILE_SIZE)](
                    self.d_pos, self.d_pos_pred,
                    self.d_curvature, self.d_particle_hashes, self.d_particle_indices,
                    self.num_x, self.num_y, self.spacing_sq,
                    threshold, self.CELL_SIZE, self.HASH_TABLE_SIZE
                )
            else:
                self.fused_curvature_hash_kernel[self.blocks_per_grid_2d, self.threads_per_block_2d](
                    self.d_pos, self.d_pos_pred,
                    self.d_curvature, self.d_particle_hashes, self.d_particle_indices,
                    self.num_x, self.num_y, self.spacing_sq,
                    threshold, self.CELL_SIZE, self.HASH_TABLE_SIZE
                )
        else:
            # [기존 경로] 별도 커널 사용 (Temporal Coherence만 사용하거나 컬링 없음)
            # [핵심 수정] 모든 설정에서 동일한 곡률 계산을 보장하여 일관성 유지
            # baseline에서도 곡률을 계산하여 temporal_coherence와 동일한 상태를 유지
            if self.use_curvature_culling:
                if self.use_temporal_coherence:
                    # Temporal Coherence 적용 (이 경로는 위에서 처리되므로 여기 도달하지 않음)
                    self.compute_update_mask_kernel[self.blocks_per_grid_2d, self.threads_per_block_2d](
                        self.d_pos, self.d_pos_cache, self.d_cache_age, self.d_update_mask,
                        self.motion_threshold, self.max_cache_age, self.num_x, self.num_y
                    )
                    
                    self.compute_curvature_selective_kernel[self.blocks_per_grid_2d, self.threads_per_block_2d](
                        self.d_pos, self.d_curvature, self.d_curvature_cache,
                        self.d_pos_cache, self.d_cache_age, self.d_update_mask,
                        self.num_x, self.num_y, self.spacing_sq
                    )
                else:
                    # 전체 곡률 계산
                    if self.use_tiled_curvature:
                        tiled_blocks_x = int(math.ceil(self.num_x / self.TILE_SIZE))
                        tiled_blocks_y = int(math.ceil(self.num_y / self.TILE_SIZE))
                        self.compute_curvature_tiled_kernel[(tiled_blocks_x, tiled_blocks_y), (self.TILE_SIZE, self.TILE_SIZE)](
                            self.d_pos, self.d_curvature, self.num_x, self.num_y, self.spacing_sq
                        )
                    else:
                        self.compute_curvature_kernel[self.blocks_per_grid_2d, self.threads_per_block_2d](
                            self.d_pos, self.d_curvature, self.num_x, self.num_y, self.spacing_sq
                        )
            else:
                # [핵심 수정] baseline이나 temporal_coherence (culling 없음)에서도 곡률을 계산
                # 이렇게 하면 모든 설정에서 동일한 곡률 값을 가지게 되어 일관성 유지
                if self.use_temporal_coherence:
                    # Temporal Coherence 적용: 선택적 곡률 계산
                    self.compute_update_mask_kernel[self.blocks_per_grid_2d, self.threads_per_block_2d](
                        self.d_pos, self.d_pos_cache, self.d_cache_age, self.d_update_mask,
                        self.motion_threshold, self.max_cache_age, self.num_x, self.num_y
                    )
                    
                    self.compute_curvature_selective_kernel[self.blocks_per_grid_2d, self.threads_per_block_2d](
                        self.d_pos, self.d_curvature, self.d_curvature_cache,
                        self.d_pos_cache, self.d_cache_age, self.d_update_mask,
                        self.num_x, self.num_y, self.spacing_sq
                    )
                else:
                    # baseline: 전체 곡률 계산 (culling은 하지 않지만 곡률은 계산)
                    if self.use_tiled_curvature:
                        tiled_blocks_x = int(math.ceil(self.num_x / self.TILE_SIZE))
                        tiled_blocks_y = int(math.ceil(self.num_y / self.TILE_SIZE))
                        self.compute_curvature_tiled_kernel[(tiled_blocks_x, tiled_blocks_y), (self.TILE_SIZE, self.TILE_SIZE)](
                            self.d_pos, self.d_curvature, self.num_x, self.num_y, self.spacing_sq
                        )
                    else:
                        self.compute_curvature_kernel[self.blocks_per_grid_2d, self.threads_per_block_2d](
                            self.d_pos, self.d_curvature, self.num_x, self.num_y, self.spacing_sq
                        )
            
            # 해시 계산 (별도 커널)
            self.compute_hash_kernel_v2[blocks_1d, threads_1d](
                self.d_pos_pred, self.d_particle_hashes, self.d_particle_indices,
                self.d_cell_start, self.d_cell_end, self.d_curvature, threshold,
                self.num_particles, self.CELL_SIZE, self.HASH_TABLE_SIZE
            )

        self._sort_particles_torch()

        self.find_cell_start_end_kernel[self.blocks_particles, self.threads_per_block](
            self.d_particle_hashes, self.d_cell_start, self.d_cell_end, self.num_particles
        )

        # Curvature threshold: use_curvature_culling이 True면 실제 임계값, 아니면 0 (모든 파티클 통과)
        curv_thresh = self.curvature_threshold if self.use_curvature_culling else 0.0
        
        # Active pair 카운터 초기화
        self.d_active_pair_count[0] = 0
        
        self.solve_self_collision_friction_kernel[self.blocks_particles, self.threads_per_block](
            self.d_pos_pred, self.d_pos, self.d_mass_inv,
            self.d_cell_start, self.d_cell_end,
            self.d_particle_indices, self.d_particle_hashes,
            self.num_particles, self.thickness, self.d_penetration,
            dt_sub, self.d_visibility, self.frame_count, self.d_debug_skip_count,
            self.d_curvature, curv_thresh,
            self.d_active_pair_count
        )

    def _step_baseline_collision(self, dt_sub):
        """
        [True Baseline] Brute-force 방식의 충돌 처리
        Spatial Hashing, Temporal Coherence, Curvature Culling 모두 사용하지 않음.
        구조화된 그리드의 이웃만 검색하는 O(n) 방식 (실제 O(n²)는 너무 느림)
        """
        self.d_penetration[:] = 0.0

        # Brute-force 충돌 검사 (구조화된 그리드 이웃만)
        self.solve_self_collision_bruteforce_kernel[self.blocks_particles, self.threads_per_block](
            self.d_pos_pred, self.d_pos, self.d_mass_inv,
            self.num_particles, self.thickness, self.d_penetration,
            dt_sub, self.num_x, self.num_y
        )

    def get_positions(self):
        return self.d_pos.copy_to_host()

    def get_penetration_depth(self):
        return self.d_penetration.copy_to_host()
    
    def cleanup(self):
        """GPU 메모리 정리"""
        del self.d_pos
        del self.d_pos_pred
        del self.d_vel
        del self.d_mass_inv
        del self.d_constraints
        del self.d_rest_lengths
        del self.d_particle_hashes
        del self.d_particle_indices
        del self.d_cell_start
        del self.d_cell_end
        del self.d_penetration
        del self.d_faces
        del self.d_normals
        del self.d_visibility
        del self.d_camera_pos
        del self.d_curvature
        del self.d_curvature_cache
        del self.d_pos_cache
        del self.d_cache_age
        del self.d_update_mask
        del self.d_debug_skip_count
        del self.d_update_counter
        del self.sphere_params
        del self.wind_vel
        for batch in self.d_color_batches:
            del batch
        gc.collect()
        cuda.current_context().memory_manager.deallocations.clear()


def run_single_benchmark(size, culling_config, total_frames=None):
    """
    단일 벤치마크 실행
    
    Returns:
        list of dict: 프레임별 측정 결과
    """
    # 기본값 처리 (모듈 로드 시점이 아닌 호출 시점의 TOTAL_FRAMES 사용)
    if total_frames is None:
        total_frames = TOTAL_FRAMES
    
    config_name, use_spatial, use_temporal, use_curvature = culling_config
    
    print(f"\n{'='*60}")
    print(f"Benchmark: Size={size}x{size}, Config={config_name}")
    print(f"  - Spatial Hashing: {use_spatial}")
    print(f"  - Temporal Coherence: {use_temporal}")
    print(f"  - Curvature Culling: {use_curvature}")
    print(f"{'='*60}")
    
    # 시뮬레이터 초기화
    sim = BenchmarkClothSimulator(
        size, size, 
        physical_width=PHYSICAL_WIDTH, 
        dt=DT, 
        substeps=SUBSTEPS,
        use_spatial_hashing=use_spatial,
        use_temporal_coherence=use_temporal,
        use_curvature_culling=use_curvature
    )
    
    # CUDA Event 생성
    start_event = cuda.event()
    stop_event = cuda.event()
    
    results = []
    
    # Warmup (처음 몇 프레임은 측정에서 제외)
    for _ in range(10):
        sim.step()
    cuda.synchronize()
    
    # 벤치마크 실행
    pbar = tqdm(range(total_frames), desc=f"[{size}x{size}] {config_name}")
    
    for frame in pbar:
        # GPU 시간 측정
        start_event.record()
        sim.step()
        stop_event.record()
        stop_event.synchronize()
        
        frame_time_ms = cuda.event_elapsed_time(start_event, stop_event)
        sort_time_ms = sim.last_sort_time * 1000.0
        
        # 통계 수집
        penetration = sim.get_penetration_depth()
        max_pen = np.max(penetration)
        avg_pen = np.mean(penetration)
        active_collisions = np.count_nonzero(penetration > 1e-6)
        
        # Active pair 카운트 읽기
        active_pairs = sim.d_active_pair_count.copy_to_host()[0]
        
        # 결과 저장 (소수점 셋째자리)
        result = {
            'frame': frame,
            'size': size,
            'config': config_name,
            'frame_time_ms': round(frame_time_ms, 3),
            'sort_time_ms': round(sort_time_ms, 3),
            'physics_time_ms': round(max(0.0, frame_time_ms - sort_time_ms), 3),
            'max_penetration': round(max_pen, 6),
            'avg_penetration': round(avg_pen, 6),
            'active_collisions': active_collisions,
            'active_pairs': active_pairs
        }
        results.append(result)
        
        # Progress bar 업데이트
        fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0
        pbar.set_postfix({
            'FPS': f'{fps:.1f}',
            'Time': f'{frame_time_ms:.2f}ms',
            'MaxPen': f'{max_pen*100:.2f}cm',
            'ActivePairs': f'{active_pairs:,}'
        })
    
    # 정리
    sim.cleanup()
    cuda.synchronize()
    gc.collect()
    
    return results


def run_full_benchmark(num_trials=None):
    """
    전체 벤치마크 실행 및 CSV 저장 (다중 시행 지원)
    
    Args:
        num_trials: 반복 실험 횟수 (기본값: NUM_TRIALS)
    """
    if num_trials is None:
        num_trials = NUM_TRIALS
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(RESULTS_DIR, f"benchmark_results_{timestamp}.csv")
    
    # CSV 헤더 (trial 컬럼 추가)
    headers = [
        'trial', 'frame', 'size', 'config', 
        'frame_time_ms', 'sort_time_ms', 'physics_time_ms',
        'max_penetration', 'avg_penetration', 'active_collisions', 'active_pairs'
    ]
    
    # CSV 파일 초기화
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
    
    print(f"\n{'='*70}")
    print(f"Starting Comprehensive Benchmark (with {num_trials} trials)")
    print(f"Output: {csv_filename}")
    print(f"Sizes: {CLOTH_SIZES}")
    print(f"Configs: {[c[0] for c in CULLING_CONFIGS]}")
    print(f"Total Frames per Config: {TOTAL_FRAMES}")
    print(f"Number of Trials: {num_trials}")
    print(f"{'='*70}\n")
    
    all_results = []
    
    for trial in range(1, num_trials + 1):
        print(f"\n{'#'*70}")
        print(f"# Trial {trial}/{num_trials}")
        print(f"{'#'*70}")
        
        for size in CLOTH_SIZES:
            for config in CULLING_CONFIGS:
                try:
                    results = run_single_benchmark(size, config, total_frames=TOTAL_FRAMES)
                    
                    # trial 정보 추가
                    for r in results:
                        r['trial'] = trial
                    
                    all_results.extend(results)
                    
                    # 중간 저장 (프로그램 중단 시 데이터 유실 방지)
                    with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=headers)
                        writer.writerows(results)
                        
                    print(f"[OK] [Trial {trial}] Saved {len(results)} frames for {size}x{size} - {config[0]}")
                    
                except Exception as e:
                    print(f"[ERROR] [Trial {trial}] Error in {size}x{size} - {config[0]}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print(f"\n{'='*70}")
    print(f"Benchmark Complete!")
    print(f"Total Results: {len(all_results)} frames across {num_trials} trials")
    print(f"Saved to: {csv_filename}")
    print(f"{'='*70}\n")
    
    return csv_filename


# 시각화 함수는 visualize.py로 이동됨
# 필요시 import: from visualize import visualize_results


if __name__ == "__main__":
    import argparse


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Cloth Simulation Benchmark')
    parser.add_argument('--mode', choices=['benchmark', 'visualize', 'both'], 
                        default='both', help='실행 모드')
    parser.add_argument('--csv', type=str, help='시각화할 CSV 파일 경로')
    parser.add_argument('--frames', type=int, default=TOTAL_FRAMES, 
                        help='테스트 프레임 수')
    parser.add_argument('--sizes', type=str, default=None,
                        help='테스트할 사이즈 (콤마로 구분, 예: 128,256,512)')
    parser.add_argument('--trials', type=int, default=NUM_TRIALS,
                        help='반복 실험 횟수 (mean±std 계산용, 기본값: 3)')
    
    args = parser.parse_args()
    
    # 사이즈 파싱
    if args.sizes:
        CLOTH_SIZES = [int(s.strip()) for s in args.sizes.split(',')]
    
    TOTAL_FRAMES = args.frames
    NUM_TRIALS = args.trials
    
    if args.mode == 'benchmark':
        csv_path = run_full_benchmark(num_trials=NUM_TRIALS)
    elif args.mode == 'visualize':
        if not args.csv:
            print("[ERROR] --csv argument is required.")
            sys.exit(1)
        from visualize import visualize_results
        visualize_results(args.csv)
    else:  # both
        csv_path = run_full_benchmark(num_trials=NUM_TRIALS)
        from visualize import visualize_results
        visualize_results(csv_path)
