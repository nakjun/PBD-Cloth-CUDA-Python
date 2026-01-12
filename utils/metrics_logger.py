import csv
import time
import os
import numpy as np

class MetricsLogger:
    def __init__(self, save_dir="experiment_logs", exp_name="cloth_sim_v1"):
        self.save_dir = save_dir
        self.exp_name = exp_name
        os.makedirs(save_dir, exist_ok=True)
        
        self.filepath = os.path.join(save_dir, f"{exp_name}_metrics.csv")
        self.data = []
        self.start_time = None
        
        # CSV 헤더 정의
        self.headers = [
            "frame", 
            "fps", 
            "physics_time_ms",    # 물리 연산 총 시간
            "collision_time_ms",  # 충돌 처리 시간 (Time Breakdown)
            "max_penetration",    # 최대 침투 깊이 (Robustness)
            "avg_penetration",    # 평균 침투 깊이
            "active_collisions"   # 충돌 쌍 개수
        ]
        
        # 파일 초기화 (덮어쓰기 방지 로직은 자네가 추가해도 좋네)
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def start_frame(self):
        """프레임 시작 시간 기록"""
        self.start_time = time.perf_counter()

    def log_frame(self, frame_idx, collision_time, max_pen, avg_pen, active_col_count):
        """
        매 프레임 끝에서 호출하여 데이터 저장
        """
        end_time = time.perf_counter()
        total_time = end_time - self.start_time
        
        fps = 1.0 / total_time if total_time > 0 else 0
        physics_ms = total_time * 1000
        collision_ms = collision_time * 1000
        
        row = [
            frame_idx,
            f"{fps:.2f}",
            f"{physics_ms:.4f}",
            f"{collision_ms:.4f}",
            f"{max_pen:.6f}",
            f"{avg_pen:.6f}",
            active_col_count
        ]
        
        self.data.append(row)
        
        # 실시간으로 파일에 씀 (프로그램이 뻗었을 때 데이터 유실 방지)
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        return fps # 콘솔 출력용으로 반환