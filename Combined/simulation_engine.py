import torch
import numpy as np
from numba import cuda
import sys
import os
import math
import argparse
import csv  # [NEW] CSV 저장을 위한 모듈

# [User Imports] 자네가 정의한 경로 설정
sys.path.append('../')

# === 추가: save_obj_with_heatmap 임포트 ===
try:
    from main import save_obj_with_heatmap
except ImportError:
    # 만약 main.py에서 임포트가 안 될 경우를 대비해 내부에 정의 (혹은 None 처리)
    # 여기서는 아래에 정의된 함수를 사용하므로 pass
    pass

# 1. 커널 Import
from cloth_cuda_optimized import compute_features_kernel, solve_self_collision_masked_kernel, solve_ground_collision_kernel
from PBD.module import predict_position_kernel, update_velocity_kernel, compute_hash_kernel, find_cell_start_end_kernel
from PBD.coloring import compute_graph_coloring
from PBD.module import solve_distance_constraint_colored_kernel 

# ------------------------------------------------------------------------------
# Helper Function: OBJ Save with Heatmap
# ------------------------------------------------------------------------------
def save_obj_with_heatmap(filename, vertices, penetrations, width, height, thickness):
    """
    [Heatmap Visualization]
    thickness: 시뮬레이션에서 사용된 파티클의 반지름 (self.thickness)
    """
    diameter = thickness * 2.0
    ignore_threshold = diameter * 0.05 
    critical_threshold = diameter * 0.3 

    with open(filename, 'w') as f:
        f.write("# Cloth Simulation Step with Calibrated Heatmap\n")
        for i, v in enumerate(vertices):
            depth = penetrations[i]
            
            ratio = 0.0
            if depth <= ignore_threshold:
                ratio = 0.0
            else:
                ratio = (depth - ignore_threshold) / (critical_threshold - ignore_threshold)
                ratio = min(max(ratio, 0.0), 1.0)
            
            r = 1.0
            g = 1.0 - ratio
            b = 1.0 - ratio
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {r:.4f} {g:.4f} {b:.4f}\n")

        for y in range(height - 1):
            for x in range(width - 1):
                idx = y * width + x + 1
                f.write(f"f {idx} {idx + width} {idx + 1}\n")
                f.write(f"f {idx + 1} {idx + width} {idx + width + 1}\n")

# ------------------------------------------------------------------------------
# Class Definition
# ------------------------------------------------------------------------------
class PowerfulClothSim:
    def __init__(self, width, height, model_path, spacing=0.1):
        """
        초강력 AI 기반 천 시뮬레이션 엔진 초기화
        """
        self.width = width
        self.height = height
        self.num_particles = width * height
        self.spacing = spacing
        
        # 물리 파라미터
        self.dt = 0.001
        self.substeps = 10 
        self.gravity = -9.8

        lift_height = 3.0 # 3미터 상공에서 시작
        
        print(f"⚡ Initializing PowerfulClothSim ({width}x{height})")
        print(f"   - Particles: {self.num_particles}")

        # 1. Host Data Setup
        pos_host = np.zeros((self.num_particles, 3), dtype=np.float32)
        # [핵심 설정]
        start_y = 1.5 # 바닥에서 1미터 정도 띄움 (너무 높으면 떨어지다 펴짐)
        
        # 주름 파라미터 (Stacking을 유도하는 가이드)
        fold_frequency = 10.0 # 주름의 빈도 (높을수록 자잘하게 접힘)
        fold_amplitude = spacing * 0.5 # 주름의 깊이 (너무 깊으면 이미 접힌 상태)

        for y in range(height):
            for x in range(width):
                idx = y * width + x
                
                # X축: 정간격 배치
                pos_x = x * spacing
                
                # Y축: 수직으로 배치하되, 공중에 띄움
                # (height가 클수록 위로 길게 뻗음)
                pos_y = (height * spacing) - (y * spacing) + start_y
                
                # [핵심] Z축: Y 높이에 따라 Sine Wave를 줌
                # 이것이 "접히는 방향"을 결정함 (지그-재그 유도)
                # y가 변함에 따라 z가 앞뒤로 흔들림
                pos_z = math.sin(y * 0.5) * fold_amplitude
                
                # (옵션) 약간 기울여서 떨어뜨리면 더 자연스러움 (Random Tilt)
                # pos_z += y * 0.01 
                
                pos_host[idx] = [pos_x, pos_y, pos_z]
        

        # for y in range(height):
        #     for x in range(width):
        #         idx = y * width + x
                
        #         # [변경 1] X축: 압축 없이 정간격 배치
        #         pos_x = x * spacing 
                
        #         # [변경 2] Y축: 공중에 띄움 (커튼처럼 수직으로 배치)
        #         # 바닥(0.0)에 닿을 때까지 떨어지도록 높이 설정
        #         pos_z = (-y * spacing) + (height * spacing) + lift_height
                
        #         # [변경 3] Z축: Sine Wave 대신 '미세한 노이즈' 추가
        #         # 완벽한 평면은 시뮬레이션에서 오히려 부자연스러움 (Buckling 유도용)
        #         # -0.01 ~ 0.01 정도의 아주 작은 난수
        #         pos_y = np.random.uniform(2.5, 3.5) 
                
        #         pos_host[idx] = [pos_x, pos_y, pos_z]
        
        # 아코디언 주름 초기화
        compression_ratio = 0.3
        # for y in range(height):
        #     for x in range(width):
        #         idx = y * width + x
        #         freq = 1.5
        #         amp = spacing * 2.0
        #         z_offset = np.sin(x * freq) * amp
        #         pos_host[idx] = [
        #             x * spacing * compression_ratio, 
        #             -y * spacing + (height * spacing), 
        #             z_offset
        #         ]

        # for y in range(height):
        #     for x in range(width):
        #         idx = y * width + x
                
        #         freq = 1.5 
        #         amp = spacing * 2.0 
        #         z_offset = np.sin(x * freq) * amp # 사인파 주름
                
        #         pos_host[idx] = [
        #             x * spacing * compression_ratio, 
        #             # [핵심] Y축 좌표를 lift_height 만큼 들어올림
        #             -y * spacing + (height * spacing) + lift_height, 
        #             z_offset
        #         ]

        # [수정] 제약 조건 생성 (Structural + Shear)
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
        
        
        # # 제약 조건 생성
        # constraints = []
        # for y in range(height):
        #     for x in range(width):
        #         idx = y * width + x
        #         if x < width - 1: constraints.append([idx, idx + 1])
        #         if y < height - 1: constraints.append([idx, idx + width])
        
        self.num_constraints = len(constraints)
        
        # 2. Graph Coloring
        print("🎨 Computing Graph Coloring...")
        color_batches_host = compute_graph_coloring(self.num_particles, constraints)
        self.d_color_batches = [cuda.to_device(batch) for batch in color_batches_host]
        
        # Rest Lengths
        constraints = np.array(constraints, dtype=np.int32)
        # rest_lengths = np.full(self.num_constraints, spacing, dtype=np.float32)
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

        # 3. GPU Memory Allocation
        self.d_pos = cuda.to_device(pos_host)
        self.d_pos_pred = cuda.device_array_like(self.d_pos)
        self.d_vel = cuda.to_device(np.zeros_like(pos_host))
        self.d_constraints = cuda.to_device(constraints)
        self.d_rest_lengths = cuda.to_device(rest_lengths)
        
        mass_inv = np.ones(self.num_particles, dtype=np.float32)
        mass_inv[0] = 0.0 
        # mass_inv[width-1] = 0.0 
        self.d_mass_inv = cuda.to_device(mass_inv)
        
        # Spatial Hashing Buffers
        # [Optimized] 100만 파티클(1024^2) 대응을 위해 해시 크기 증설 (약 300만)
        self.HASH_SIZE = 2999999 
        self.d_particle_hashes = cuda.device_array(self.num_particles, dtype=np.int32)
        self.d_particle_indices = cuda.device_array(self.num_particles, dtype=np.int32)
        self.d_cell_start = cuda.device_array(self.HASH_SIZE, dtype=np.int32)
        self.d_cell_end = cuda.device_array(self.HASH_SIZE, dtype=np.int32)
        self.thickness = spacing * 0.3
        self.d_penetration = cuda.device_array(self.num_particles, dtype=np.float32)

        # CUDA Config
        self.threads = 256
        self.blocks = (self.num_particles + 255) // 256
        
        # 4. AI & Zero-Copy Setup
        print(f"🧠 Loading AI Brain from {model_path}...")
        
        if os.path.exists(model_path):
            self.ai_model = torch.nn.Sequential(
                torch.nn.Linear(4, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
                torch.nn.Sigmoid()
            ).cuda()
            
            checkpoint = torch.load(model_path, map_location='cuda')
            new_state_dict = {k.replace("net.", ""): v for k, v in checkpoint.items()}
            self.ai_model.load_state_dict(new_state_dict, strict=False)
            self.ai_model.eval()
            
            try:
                self.ai_model = torch.compile(self.ai_model)
                print("🚀 PyTorch 2.0 Compiled Model Activated!")
            except:
                pass
        else:
            raise FileNotFoundError("Model not found!")

        self.d_features = cuda.device_array((self.num_particles, 4), dtype=np.float32)
        self.d_risk_mask = cuda.device_array(self.num_particles, dtype=np.float32)
        
        self.frame_count = 0
        self.ai_interval = 10 
        
        print("✅ Simulation Engine Ready. Let's Rock.")

    def _numba_to_torch(self, numba_array):
        return torch.as_tensor(numba_array, device='cuda')

    def _run_ai_culling(self):
        """
        AI 기반 충돌 가지치기 (Frame 당 1회 수행 권장)
        """
        # 1. Feature Extraction (GPU Kernel)
        # CPU 개입 없이 GPU 안에서만 데이터 이동
        compute_features_kernel[self.blocks, self.threads](
            self.d_pos, self.d_vel, self.d_features,
            self.width, self.height, self.spacing
        )
        # 2. Inference (Zero-Copy)
        input_tensor = self._numba_to_torch(self.d_features)
        with torch.no_grad():
            # (N, 4) -> Model -> (N, 1)
            probs = self.ai_model(input_tensor)
            # Thresholding (0.5)
            # (N, 1) -> (N, )
            mask_tensor = (probs > 0.5).float().squeeze()
            # 3. Write back to Numba Buffer [FIXED]
            # PyTorch Tensor -> Numba Wrapper 변환
            # (mask_tensor가 메모리상 연속적이지 않을 수 있으므로 contiguous() 호출 필수)
            cuda_mask_view = cuda.as_cuda_array(mask_tensor.contiguous())
            # [핵심 수정] 슬라이싱 대입을 이용한 Device-to-Device Copy
            self.d_risk_mask[:] = cuda_mask_view
            # self.d_risk_mask[:] = 1.0


    
    # def _run_ai_culling(self):
    #     compute_features_kernel[self.blocks, self.threads](
    #         self.d_pos, self.d_vel, self.d_features,
    #         self.width, self.height, self.spacing
    #     )
        
    #     input_tensor = self._numba_to_torch(self.d_features)
        
    #     with torch.no_grad():
    #         probs = self.ai_model(input_tensor)
    #         mask_tensor = (probs > 0.5).float().squeeze() 
    #         cuda_mask_view = cuda.as_cuda_array(mask_tensor.contiguous())
    #         self.d_risk_mask[:] = cuda_mask_view

    def _sort_particles_torch(self):
        hashes_torch = self._numba_to_torch(self.d_particle_hashes)
        indices_torch = self._numba_to_torch(self.d_particle_indices)
        
        sorted_indices = torch.argsort(hashes_torch)
        
        hashes_sorted = hashes_torch[sorted_indices]
        indices_sorted = indices_torch[sorted_indices]
        
        self.d_particle_hashes[:] = cuda.as_cuda_array(hashes_sorted.contiguous())
        self.d_particle_indices[:] = cuda.as_cuda_array(indices_sorted.contiguous())

    def step(self):
        dt_sub = self.dt / self.substeps
        
        # [Step 1] AI Culling (Interleaved)
        if self.frame_count % self.ai_interval == 0:
            self._run_ai_culling()
        
        self.frame_count += 1
        
        # [Step 2] PBD Substeps
        for _ in range(self.substeps):
            predict_position_kernel[self.blocks, self.threads](
                self.d_pos, self.d_vel, self.d_pos_pred, self.d_mass_inv, 
                dt_sub, self.gravity, self.num_particles
            )
            
            for d_batch in self.d_color_batches:
                blocks_c = (d_batch.shape[0] + 255) // 256
                solve_distance_constraint_colored_kernel[blocks_c, 256](
                    self.d_pos_pred, self.d_mass_inv, self.d_constraints, 
                    self.d_rest_lengths, d_batch, dt_sub, 0.8
                )
            
            self.d_cell_start[:] = -1
            self.d_cell_end[:] = -1
            compute_hash_kernel[self.blocks, self.threads](
                self.d_pos_pred, self.d_particle_hashes, self.d_particle_indices, self.num_particles
            )
            
            self._sort_particles_torch()
            
            find_cell_start_end_kernel[self.blocks, self.threads](
                self.d_particle_hashes, self.d_cell_start, self.d_cell_end, self.num_particles
            )

            solve_ground_collision_kernel[self.blocks, self.threads](
                self.d_pos_pred, self.d_pos, self.d_vel, 
                self.num_particles, 0.0, 0.5
            )

            
            solve_self_collision_masked_kernel[self.blocks, self.threads](
                self.d_pos_pred, self.d_mass_inv,
                self.d_cell_start, self.d_cell_end,
                self.d_particle_indices, self.d_particle_hashes,
                self.d_risk_mask,
                self.num_particles, self.thickness, self.d_penetration
            )
            
            update_velocity_kernel[self.blocks, self.threads](
                self.d_pos, self.d_vel, self.d_pos_pred, dt_sub, self.num_particles
            )

    # --- Data Access ---
    def get_positions(self):
        return self.d_pos.copy_to_host()
    
    def get_risk_mask(self):
        return self.d_risk_mask.copy_to_host()

    def get_penetrations(self):
        return self.d_penetration.copy_to_host()

# ------------------------------------------------------------------------------
# Main Logic with Arguments
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Powerful AI Cloth Simulation Engine")
    parser.add_argument("--type", type=int, default=1, 
                        help="Mode 1: Single FPS Benchmark, Mode 2: Extract OBJ, Mode 3: Grid Search Benchmark")
    args = parser.parse_args()

    # 모델 경로
    MODEL_PATH = "../MLP/best_model_v2.pth"

    # [Type 3가 아닐 때만 기본 1024x1024 생성]
    # Grid Search 때는 해상도를 바꿔가며 생성해야 하므로 여기서는 생성하지 않거나, 생성 후 무시함.
    if args.type != 3:
        # 1. 초기화 (기본)
        sim = PowerfulClothSim(128, 128, MODEL_PATH, spacing=0.1)
        print("🔥 Warming up GPU...")
        for _ in range(10): sim.step()
        torch.cuda.synchronize()

    # ==========================================
    # TYPE 1: Average FPS Benchmark (Single)
    # ==========================================
    if args.type == 1:
        print("\n[MODE 1] Starting FPS Benchmark (1024x1024)...")
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        total_time_sum = 0.0
        avg_active_ratio = 0.0
        
        print("⏱️  Profiling 100 frames...")
        
        for i in range(100):
            start_event.record()
            sim.step()
            end_event.record()
            torch.cuda.synchronize()
            
            frame_time = start_event.elapsed_time(end_event)
            total_time_sum += frame_time
            
        avg_fps = 1000.0 / (total_time_sum / 100)
        print("="*40)
        print(f"🚀 Final Result:")
        print(f"   - Average FPS: {avg_fps:.2f}")
        print(f"   - Avg Active Ratio: {(avg_active_ratio/100)*100:.1f}%")
        print("="*40)

    # ==========================================
    # TYPE 2: Extract OBJ Files
    # ==========================================
    elif args.type == 2:
        print("\n[MODE 2] Extracting OBJ files with Heatmap...")
        
        output_dir = "extracted_objs_baseline_v3"
        os.makedirs(output_dir, exist_ok=True)
        
        TOTAL_FRAMES = 3000
        SAVE_INTERVAL = 10
        
        print(f"📂 Output Directory: {output_dir}")

        for i in range(TOTAL_FRAMES):
            sim.step()
            
            if i % SAVE_INTERVAL == 0:
                pos = sim.get_positions()
                pen = sim.get_penetrations()
                
                filename = os.path.join(output_dir, f"cloth_{i:04d}.obj")
                save_obj_with_heatmap(
                    filename, pos, pen, sim.width, sim.height, sim.thickness
                )
                print(f"   💾 Saved: {filename}", end='\r')
        
        print(f"\n✅ Extraction Complete! Check '{output_dir}' folder.")

    # ==========================================
    # TYPE 3: Grid Search Benchmark (CSV Save)
    # ==========================================
    elif args.type == 3:
        print("\n[MODE 3] Starting Grid Search Benchmark...")
        
        # [Grid Search Settings]
        # Cloth Simulation에서 자주 사용되는 해상도 (2의 제곱수)
        resolutions = [64, 128, 256, 512, 1024] 
        csv_filename = "grid_search_results.csv"
        
        # CSV 파일 초기화
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Resolution", "Particles", "Average_FPS", "Average_Active_Ratio_Percent"])

        print(f"📋 Resolutions to test: {resolutions}")
        print(f"💾 Results will be saved to: {csv_filename}")

        for res in resolutions:
            print("\n" + "-"*50)
            print(f"🧪 Testing Resolution: {res} x {res}")
            print("-"*50)

            # 메모리 정리 (이전 시뮬레이션 데이터 해제)
            if 'sim' in locals():
                del sim
            torch.cuda.empty_cache()

            try:
                # 1. 시뮬레이션 인스턴스 생성
                # 여기서는 비교 통제를 위해 spacing 고정
                sim = PowerfulClothSim(res, res, MODEL_PATH, spacing=0.1)
                
                # Warmup
                print("   🔥 Warming up...")
                for _ in range(10): sim.step()
                torch.cuda.synchronize()

                # 2. 벤치마킹
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                total_time_sum = 0.0
                avg_active_ratio = 0.0
                TEST_FRAMES = 100

                print(f"   ⏱️  Profiling {TEST_FRAMES} frames...")
                
                for i in range(TEST_FRAMES):
                    start_event.record()
                    sim.step()
                    end_event.record()
                    torch.cuda.synchronize()
                    
                    frame_time = start_event.elapsed_time(end_event)
                    total_time_sum += frame_time
                    
                    # Benchmark 과정에서는 mask copy를 하지 않음
                    # 즉, active ratio 측정 코드도 생략
                    # (아래 Block 완전히 제거)
                    # if i % 10 == 0:
                    #     mask = sim.d_risk_mask.copy_to_host()
                    #     active_count = np.sum(mask > 0.5)
                    #     avg_active_ratio += (active_count / sim.num_particles)

                # 결과 계산
                avg_fps = 1000.0 / (total_time_sum / TEST_FRAMES)
                # 실제로 active ratio 측정을 생략했으므로 0.0으로 설정
                final_active_ratio = 0.0

                print(f"   ✅ Result: {avg_fps:.2f} FPS | Active: {final_active_ratio:.2f}%")

                # CSV 저장
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([f"{res}x{res}", sim.num_particles, f"{avg_fps:.2f}", f"{final_active_ratio:.2f}"])

            except Exception as e:
                print(f"   ❌ Error at {res}x{res}: {e}")
                # 에러나면 CSV에 에러 기록
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([f"{res}x{res}", "ERROR", "0.0", "0.0"])

        print("\n🎉 Grid Search Complete! Data saved to CSV.")