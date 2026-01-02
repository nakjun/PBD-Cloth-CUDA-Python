import os
from Cloth.cloth import ClothSimulator
from tqdm import tqdm

# 간단한 OBJ 파일 저장 함수
def save_obj(filename, vertices, width, height):
    with open(filename, 'w') as f:
        f.write("# Cloth Simulation Step\n")

        # 정점(Vertex) 쓰기
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")

        # 면(Face) 쓰기 (인덱스는 1부터 시작)
        for y in range(height - 1):
            for x in range(width - 1):
                idx = y * width + x + 1
                # Triangle 1
                f.write(f"f {idx} {idx + width} {idx + 1}\n")
                # Triangle 2
                f.write(f"f {idx + 1} {idx + width} {idx + width + 1}\n")

def main_headless():
    print("Initialize Headless Simulation...")

    # 32x32 해상도
    width, height = 256, 256
    sim = ClothSimulator(width, height, spacing=0.1)

    # 출력 폴더 생성
    output_dir = "output_frames_self_collision_v3"
    os.makedirs(output_dir, exist_ok=True)

    total_frames = 10000
    print(f"Start simulation for {total_frames} frames...")

    for frame in tqdm(range(total_frames), desc="Simulating frames"):
        sim.step()

        # 데이터 가져오기
        positions = sim.get_positions()

        # 로그 출력 (중앙 하단 점의 위치를 추적하여 잘 떨어지는지 확인)
        center_bottom_idx = (height - 1) * width + (width // 2)
        curr_y = positions[center_bottom_idx][1]

        # print(f"[Frame {frame:03d}] Center-Bottom Y: {curr_y:.4f}")

        # 10프레임마다 OBJ 파일 저장 (나중에 MeshLab/Blender로 확인)
        if frame % 10 == 0:
            save_obj(f"{output_dir}/cloth_{frame:03d}.obj", positions, width, height)
            # print(f"  -> Saved {output_dir}/cloth_{frame:03d}.obj")

    print("Simulation Finished.")

if __name__ == "__main__":
    # main() 대신 main_headless() 실행
    main_headless()