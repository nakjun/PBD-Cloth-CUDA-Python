import sys
import os
import numpy as np
import math
import time
from tqdm import tqdm
from numba import cuda # [필수] CUDA Event 사용을 위해 필요

# 프로젝트 루트 경로 추가 (모듈 import용)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# PBD 모듈 및 유틸리티 import
from PBD.cloth import ClothSimulator
from PBD.render_utils import ClothRenderer
from utils.metrics_logger import MetricsLogger

# ============================================================
# [실험 모드 설정]
# "RENDER": 시각화 이미지 저장
# "BENCHMARK": 렌더링 없이 FPS/성능 측정 (성능 검증용)
# ============================================================
# EXP_MODE = "RENDER"
EXP_MODE = "BENCHMARK"
# ============================================================

# 1. 환경 설정
SIZE = 1024
WIDTH = SIZE
HEIGHT = SIZE
PHYSICAL_WIDTH = 12.0

# 2. 시간 설정
DT = 0.01
# 해상도가 높을수록 substeps를 늘려야 안정적입니다. (예: 512 해상도에서는 15~20 권장)
SUBSTEPS = 15
TOTAL_FRAMES = 5000

# 실험 이름 설정
if EXP_MODE == "RENDER":
    EXP_NAME = f"view_culling_v3_render_{SIZE}"
elif EXP_MODE == "BENCHMARK":
    EXP_NAME = f"view_culling_v3_bench_{SIZE}"

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
# [수정] 정확한 GPU 시간 측정을 위한 CUDA Event 객체 생성
start_event = None
stop_event = None

if EXP_MODE == "RENDER":
    RENDER_DIR = os.path.join(BASE_DIR, "frames")
    renderer = ClothRenderer(WIDTH, HEIGHT, save_dir=RENDER_DIR)
    print(f"[Info] Renderer initialized. Saving frames to: {RENDER_DIR}")
    TOTAL_FRAMES = 2500

elif EXP_MODE == "BENCHMARK":
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    logger = MetricsLogger(save_dir=LOG_DIR, exp_name=EXP_NAME)
    print(f"[Info] Benchmark Logger initialized. Saving logs to: {LOG_DIR}")
    TOTAL_FRAMES = 50
    
    # [수정] 벤치마크 모드에서만 이벤트 초기화
    start_event = cuda.event()
    stop_event = cuda.event()


# ============================================================
# Main Simulation Loop
# ============================================================
print(f"Start simulation loop...")
# [수정] tqdm 객체를 변수 pbar에 할당하여 루프 내에서 업데이트 가능하게 함
pbar = tqdm(range(TOTAL_FRAMES), desc=f"Simulating ({EXP_MODE})")

# [수정] 밀리초(ms) 단위로 누적 시간 계산
total_pure_sim_time_ms = 0.0

# [수정] 루프 변수명을 'frame'으로 통일하고 pbar를 순회
for frame in pbar:
    
    if EXP_MODE == "BENCHMARK":
        logger.start_frame()

    # ---------------------------------------------------------
    # [Physics Step] 물리 시뮬레이션 수행 및 시간 측정 (핵심 수정)
    # ---------------------------------------------------------
    frame_total_time_ms = 0.0
    sort_time_ms = 0.0
    
    if EXP_MODE == "BENCHMARK":
        # [중요] 비동기 GPU 실행을 정확히 측정하기 위해 CUDA Event 사용
        
        # 1. 시작 타임스탬프 기록
        start_event.record()
        
        # 2. 커널 실행 (비동기)
        sim.step()
        
        # 3. 종료 타임스탬프 기록
        stop_event.record()
        
        # 4. [핵심] GPU가 stop_event 지점까지 작업을 마칠 때까지 CPU 대기
        stop_event.synchronize()
        
        # 5. 정확한 경과 시간 계산 (단위: 밀리초 ms)
        frame_total_time_ms = cuda.event_elapsed_time(start_event, stop_event)

        # cloth.py 내부에서 측정된 정렬 시간 가져오기 (초 단위라고 가정하고 ms로 변환)
        # 만약 cloth.py 내부도 cuda event로 ms를 측정한다면 * 1000을 제거하세요.
        sort_time_ms = sim.last_sort_time * 1000.0
        
        # 순수 물리 계산 시간 = 전체 GPU 시간 - 정렬 시간
        # (주의: 정렬 방식에 따라 이 계산이 음수가 나오거나 부정확할 수 있음. 
        # 가장 좋은 건 cloth.py 내부에서 정렬을 제외한 구간만 CUDA Event로 감싸는 것입니다.)
        pure_sim_time_ms = max(0.0, frame_total_time_ms - sort_time_ms)
        total_pure_sim_time_ms += pure_sim_time_ms
        
    else:
        # RENDER 모드에서는 정밀한 측정 불필요
        sim.step()

    # ---------------------------------------------------------
    # [Data Retrieval] GPU -> CPU 데이터 가져오기
    # ---------------------------------------------------------
    pos = sim.get_positions() # (N, 3)
    penetration = sim.get_penetration_depth() # (N,)

    # ---------------------------------------------------------
    # 모드별 동작 분기
    # ---------------------------------------------------------
    if EXP_MODE == "RENDER":
        # 5프레임마다 저장
        if frame % 5 == 0:
            sphere_params = sim.sphere_params.copy_to_host()
            renderer.render_frame(
                pos, 
                penetration,
                frame,
                mode='visual',
                sphere_params=sphere_params
            )

    elif EXP_MODE == "BENCHMARK":
        # 통계 계산
        max_pen = np.max(penetration)
        avg_pen = np.mean(penetration)
        active_collisions = np.count_nonzero(penetration > 1e-6)

        # 로그 기록 (ms 단위 시간 전달)
        logger.log_frame(
            frame_idx=frame,
            collision_time=frame_total_time_ms, 
            max_pen=max_pen,
            avg_pen=avg_pen,
            active_col_count=active_collisions
        )
        
        # [수정] FPS 및 상태창 업데이트 로직 개선
        # 현재까지의 평균 FPS 계산 (ms를 초로 변환)
        elapsed_seconds = total_pure_sim_time_ms / 1000.0
        avg_pure_fps = (frame + 1) / elapsed_seconds if elapsed_seconds > 0 else 0
        
        # [수정] tqdm 설명창 업데이트 (올바른 변수명 사용)
        pbar.set_description(
            f"FPS(Pure Avg)={avg_pure_fps:.1f} | "
            f"Time(Total)={frame_total_time_ms:.2f}ms | "
            f"Time(XPBD)={pure_sim_time_ms:.2f}ms | "
            f"Time(Sort)={sort_time_ms:.2f}ms | "
            f"MaxPen={max_pen*100:.2f}cm"
        )

# ============================================================
# 실험 종료
# ============================================================
# 마지막으로 남은 GPU 작업이 있다면 대기
cuda.synchronize()
print(f"\n✅ Simulation Finished!! : {EXP_NAME}")

if EXP_MODE == "RENDER":
    print(f"Frames saved in: {renderer.save_dir}")
elif EXP_MODE == "BENCHMARK":
    # 최종 결과 저장 (metrics.csv)
    # logger.save_metrics()
    print(f"Metrics saved in: {logger.filepath}")