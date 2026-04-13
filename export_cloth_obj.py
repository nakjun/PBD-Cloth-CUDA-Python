"""
Cloth OBJ Exporter for Blender Rendering

- 512x512, 1024x1024 시뮬레이션의 중간 프레임을 OBJ로 저장
- Blender에서 정성적(Qualitative) 렌더링용으로 사용
"""

import argparse
import os
from typing import List

import numpy as np

from PBD.cloth import ClothSimulator


def parse_sizes(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_frames(raw: str) -> List[int]:
    frames = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    return [f for f in frames if f > 0]


def configure_simulator(sim: ClothSimulator, config: str) -> None:
    config = config.lower()
    if config == "baseline":
        sim.use_curvature_culling = False
        sim.use_temporal_coherence = False
    elif config == "curvature":
        sim.use_curvature_culling = True
        sim.use_temporal_coherence = False
    elif config == "full":
        sim.use_curvature_culling = True
        sim.use_temporal_coherence = True
    else:
        raise ValueError(f"Unsupported config: {config}")


def write_obj(path: str, positions: np.ndarray, num_x: int, num_y: int) -> None:
    """
    현재 프레임 cloth mesh를 OBJ로 저장합니다.
    - v: 파티클 위치
    - f: 규칙 격자 기반 삼각형 토폴로지
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Cloth mesh exported for Blender\n")
        f.write(f"# grid: {num_x} x {num_y}\n")
        f.write(f"# vertices: {positions.shape[0]}\n")
        f.write(f"# triangles: {(num_x - 1) * (num_y - 1) * 2}\n")

        # OBJ vertex
        for p in positions:
            f.write(f"v {p[0]:.7f} {p[1]:.7f} {p[2]:.7f}\n")

        # OBJ face (1-based index)
        for y in range(num_y - 1):
            row = y * num_x
            next_row = (y + 1) * num_x
            for x in range(num_x - 1):
                i0 = row + x + 1
                i1 = row + x + 2
                i2 = next_row + x + 1
                i3 = next_row + x + 2
                f.write(f"f {i0} {i1} {i3}\n")
                f.write(f"f {i0} {i3} {i2}\n")


def export_for_size(
    size: int,
    frames_to_capture: List[int],
    output_root: str,
    config: str,
    physical_width: float,
    dt: float,
    substeps: int,
) -> None:
    sim = ClothSimulator(size, size, physical_width=physical_width, dt=dt, substeps=substeps)
    configure_simulator(sim, config)

    size_dir = os.path.join(output_root, f"{size}x{size}", config)
    os.makedirs(size_dir, exist_ok=True)

    max_frame = max(frames_to_capture)
    capture_set = set(frames_to_capture)

    print("=" * 70)
    print(f"[Export] size={size}x{size}, config={config}, total_frames={max_frame}")
    print(f"[Export] capture frames: {frames_to_capture}")
    print("=" * 70)

    for frame in range(1, max_frame + 1):
        sim.step()
        if frame in capture_set:
            pos = sim.get_positions()
            out_path = os.path.join(size_dir, f"cloth_frame_{frame:04d}.obj")
            write_obj(out_path, pos, sim.num_x, sim.num_y)
            print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cloth simulation frames to OBJ for Blender.")
    parser.add_argument(
        "--sizes",
        type=str,
        default="512,1024",
        help="Comma-separated cloth resolutions (default: 512,1024)",
    )
    parser.add_argument(
        "--capture-frames",
        type=str,
        default="1, 100,200,300, 400,500",
        help="Comma-separated frame numbers to export (default: 0, 100,200,300, 400,500)",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["baseline", "curvature", "full"],
        default="curvature",
        help="Simulation config to render (default: curvature)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="blender_exports",
        help="Output directory root (default: blender_exports)",
    )
    parser.add_argument("--physical-width", type=float, default=10.0, help="Physical cloth width (m)")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation dt")
    parser.add_argument("--substeps", type=int, default=10, help="Simulation substeps")
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    frames = parse_frames(args.capture_frames)
    if not sizes:
        raise ValueError("No valid size in --sizes")
    if not frames:
        raise ValueError("No valid frame in --capture-frames")

    os.makedirs(args.output_dir, exist_ok=True)

    for size in sizes:
        export_for_size(
            size=size,
            frames_to_capture=frames,
            output_root=args.output_dir,
            config=args.config,
            physical_width=args.physical_width,
            dt=args.dt,
            substeps=args.substeps,
        )

    print("\nDone. Import exported OBJ files in Blender for qualitative rendering.")


if __name__ == "__main__":
    main()

