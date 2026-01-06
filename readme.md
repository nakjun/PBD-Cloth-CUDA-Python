# Efficient Cloth Simulation with MLP-based Self-collision Masking Method

## Setting
```
git clone https://github.com/nakjun/PBD-Cloth-CUDA-Python.git
cd PBD-Cloth-CUDA-Python
pip install -r requirements.txt # venv 생성 후 설치 추천 / torch의 CUDA 버전은 본인 GPU와 맞추기

# MLP-Cloth 모델 테스트
cd Combined
# Performance Check / 5,000 frame동안의 FPS 계산
python .\simulation_engine.py --type 1

# OBJ Extract / 5,000 frame동안의 obj export for Blender Rendering
python .\simulation_engine.py --type 2

# Benchmark / Cloth Size를 늘리며 Benchmark test 수행
python .\simulation_engine.py --type 3
```

## Main Concept(Workflow)
![workflow](./workflow.jpg)

## Simulation Results
![simulation_results](./simulation_results.gif)