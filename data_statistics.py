import numpy as np
import glob
import os

# 데이터 경로 (v2 폴더 등)
DATA_DIRS = ["./dataset_curtain_128", "./dataset_flag_128", "./dataset_pin_128"]

vel_list = []
geo_list = []

print("📊 Loading Data Statistics...")
for d_dir in DATA_DIRS:
    files = glob.glob(os.path.join(d_dir, "*.npz"))
    for f in files:
        data = np.load(f)
        vel_list.append(data['vel'])
        geo_list.append(data['geo'].reshape(-1, 1))

# 전체 데이터 병합
all_vel = np.vstack(vel_list)
all_geo = np.vstack(geo_list)

print("="*40)
print(f"Vel Mean: {np.mean(all_vel):.6f} | Std: {np.std(all_vel):.6f}")
print(f"Vel Min:  {np.min(all_vel):.6f} | Max: {np.max(all_vel):.6f}")
print("-" * 40)
print(f"Geo Mean: {np.mean(all_geo):.6f} | Std: {np.std(all_geo):.6f}")
print(f"Geo Min:  {np.min(all_geo):.6f} | Max: {np.max(all_geo):.6f}")
print("="*40)