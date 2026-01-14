import sys
import os
import numpy as np
import math
import time
from tqdm import tqdm
from numba import cuda

# 프로젝트 루트 경로 추가 (모듈 import용)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# PBD 모듈 및 유틸리티 import
from PBD.cloth import ClothSimulator
from PBD.render_utils import ClothRenderer
# save_obj_with_heatmap은 RENDER 모드에서 renderer.render_frame으로 대체됨
from utils.metrics_logger import MetricsLogger # 성능 측정 로거

# ============================================================
# [실험 모드 설정]
# "RENDER": 시각화 이미지 저장 (기존의 [B] 시각화용 OBJ 저장 역할 대체)
# "BENCHMARK": 렌더링 없이 FPS/성능 측정 및 CSV 저장 (성능 검증용)
# ============================================================
# EXP_MODE = "RENDER"
EXP_MODE = "BENCHMARK"
# ============================================================

# 1. 환경 설정 (기존 main_data_collection의 설정 계승)
SIZE = 1024
WIDTH = SIZE
HEIGHT = SIZE
PHYSICAL_WIDTH = 12.0 # 기존 설정 유지

# 2. 시간 설정
DT = 0.01
SUBSTEPS = 15
TOTAL_FRAMES = 5000 # 기존 설정 유지 (충분한 데이터 확보)

# 실험 이름 설정 (저장 폴더명 등으로 사용)
if EXP_MODE == "RENDER":
    EXP_NAME = f"view_culling_v1_render_{SIZE}" # 기존 저장 폴더명 반영
elif EXP_MODE == "BENCHMARK":
    EXP_NAME = f"view_culling_v1_bench_{SIZE}"

print(f"=== Cloth Simulation: Data Collection & Benchmark ===")
print(f"Mode: {EXP_MODE} | Resolution: {WIDTH}x{HEIGHT}")
print(f"Physical Width: {PHYSICAL_WIDTH}m | Substeps: {SUBSTEPS}")
print(f"Total Frames: {TOTAL_FRAMES}")
print(f"Experiment Name: {EXP_NAME}")
print("=====================================================")

# 디렉토리 설정
BASE_DIR = os.path.join(current_dir, f"experiment_results/{EXP_NAME}")
os.makedirs(BASE_DIR, exist_ok=True)

# 3. 모듈 초기화
print("🎓 Initialize Simulation...")
sim = ClothSimulator(WIDTH, HEIGHT, physical_width=PHYSICAL_WIDTH, dt=DT, substeps=SUBSTEPS)

renderer = None
logger = None

if EXP_MODE == "RENDER":
    # 렌더링 모드: Renderer 초기화 (기존 ClothRenderer 사용)
    RENDER_DIR = os.path.join(BASE_DIR, "frames")
    print(RENDER_DIR)
    # 기존 코드의 save_dir="view_culling_v1"을 RENDER_DIR로 대체
    renderer = ClothRenderer(WIDTH, HEIGHT, save_dir=RENDER_DIR)
    print(f"[Info] Renderer initialized. Saving frames to: {RENDER_DIR}")

    TOTAL_FRAMES = 2500

elif EXP_MODE == "BENCHMARK":
    # 벤치마크 모드: Logger 초기화
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    logger = MetricsLogger(save_dir=LOG_DIR, exp_name=EXP_NAME)
    print(f"[Info] Benchmark Logger initialized. Saving logs to: {LOG_DIR}")

    TOTAL_FRAMES = 500


# ============================================================
# Main Simulation Loop
# ============================================================
print(f"Start simulation loop...")
loop_range = tqdm(range(TOTAL_FRAMES), desc=f"Simulating ({EXP_MODE})")

for frame in loop_range:
    
    # [BENCHMARK] 프레임 시작 시간 기록
    if EXP_MODE == "BENCHMARK":
        logger.start_frame()

    # ---------------------------------------------------------
    # [Physics Step] 물리 시뮬레이션 수행
    # ---------------------------------------------------------
    phys_start = time.perf_counter()
    sim.step()
    phys_end = time.perf_counter()
    frame_physics_time = phys_end - phys_start

    # ---------------------------------------------------------
    # [Data Retrieval] GPU -> CPU 데이터 가져오기 (중요)
    # ---------------------------------------------------------
    # RENDER 모드에서는 시각화에, BENCHMARK 모드에서는 통계 계산에 사용됨
    # ClothSimulator에 해당 메서드들이 구현되어 있어야 함
    pos = sim.get_positions()           # (N, 3)
    
    # NOTE: vel, geo_feature는 현재 RENDER/BENCHMARK 모드에서 직접 사용되지 않지만,
    # 추후 [A] AI 학습용 데이터 저장 기능 부활 시 필요함.
    # vel = sim.get_velocities()        # (N, 3)
    # geo_feature = sim.get_compression_feature(pos) # (N, 1)
    
    penetration = sim.get_penetration_depth() # (N,) : 정답 라벨/통계용

    # ---------------------------------------------------------
    # 모드별 동작 분기
    # ---------------------------------------------------------
    if EXP_MODE == "RENDER":
        # [RENDER MODE] 이미지 저장 (기존 [B] 시각화용 OBJ 저장 대체)
        
        # 5프레임마다 저장 (기존 로직 유지)
        if frame % 5 == 0:
            sphere_params = sim.sphere_params.copy_to_host()
            
            renderer.render_frame(
                pos, 
                penetration, # 히트맵 데이터 (mode='visual'에서는 무시됨)
                frame,
                mode='visual', # 단색 렌더링 모드
                sphere_params=sphere_params
            )

    elif EXP_MODE == "BENCHMARK":
        # [BENCHMARK MODE] 성능 지표 기록
        
        # 통계 계산
        max_pen = np.max(penetration)
        avg_pen = np.mean(penetration)
        # 활성 충돌 파티클 수 (침투 깊이가 0보다 큰 경우)
        active_collisions = np.count_nonzero(penetration > 1e-6)

        # 로그 기록
        # TODO: collision_time을 정확히 측정하려면 ClothSimulator 내부 수정 필요.
        # 현재는 전체 physics_time을 넘김.
        fps = logger.log_frame(
            frame_idx=frame,
            collision_time=frame_physics_time, 
            max_pen=max_pen,
            avg_pen=avg_pen,
            active_col_count=active_collisions
        )
        
        # 진행바에 정보 표시
        # frame_physics_time(초 단위)에 1000을 곱해 ms 단위로 표시
        loop_range.set_postfix({
            "Total FPS": f"{fps:.1f}", 
            "MaxPen": f"{max_pen*100:.2f}cm", 
            "Simulation Time": f"{frame_physics_time * 1000:.2f}ms"
        })

    # ---------------------------------------------------------
    # [TODO: A] AI 학습용 데이터 저장 (.npz)
    # ---------------------------------------------------------
    # 필요 시 여기에 np.savez_compressed 코드 추가 (이전 코드 참조)


# ============================================================
# 실험 종료
# ============================================================
cuda.synchronize()
print(f"\n✅ Simulation Finished!! : {EXP_NAME}")
if EXP_MODE == "RENDER":
    print(f"Frames saved in: {renderer.save_dir}")
elif EXP_MODE == "BENCHMARK":
    print(f"Metrics saved in: {logger.filepath}")