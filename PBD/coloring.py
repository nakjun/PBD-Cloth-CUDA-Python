import numpy as np

def compute_graph_coloring(num_particles, constraints):
    """
    Greedy Graph Coloring Algorithm for PBD Constraints.
    Returns: A list of arrays, where each array contains indices of constraints
             that can be solved in parallel (independent sets).
    """
    num_constraints = len(constraints)
    
    # 1. 각 파티클이 어떤 제약조건들에 연결되어 있는지 파악 (Adjacency List)
    particle_to_constraints = [[] for _ in range(num_particles)]
    for c_idx, (p1, p2) in enumerate(constraints):
        particle_to_constraints[p1].append(c_idx)
        particle_to_constraints[p2].append(c_idx)
    
    # 2. 제약조건 별 색상 할당 (초기화: -1)
    constraint_colors = [-1] * num_constraints
    num_colors = 0
    
    # 3. Greedy Coloring
    for c_idx in range(num_constraints):
        # 현재 제약조건(스프링)에 연결된 두 점(p1, p2)을 찾음
        p1, p2 = constraints[c_idx]
        
        # 나와 점을 공유하는 이웃 제약조건들이 사용 중인 색상들을 수집
        neighbor_colors = set()
        
        # p1에 연결된 다른 제약조건들 확인
        for neighbor_c_idx in particle_to_constraints[p1]:
            if neighbor_c_idx != c_idx and constraint_colors[neighbor_c_idx] != -1:
                neighbor_colors.add(constraint_colors[neighbor_c_idx])
        
        # p2에 연결된 다른 제약조건들 확인
        for neighbor_c_idx in particle_to_constraints[p2]:
            if neighbor_c_idx != c_idx and constraint_colors[neighbor_c_idx] != -1:
                neighbor_colors.add(constraint_colors[neighbor_c_idx])
        
        # 사용되지 않은 가장 낮은 색상 번호(0부터 시작)를 찾아서 할당
        color = 0
        while color in neighbor_colors:
            color += 1
        
        constraint_colors[c_idx] = color
        num_colors = max(num_colors, color + 1)
    
    # 4. 같은 색상끼리 묶어서 배치(Batch) 생성
    batches = [[] for _ in range(num_colors)]
    for c_idx, color in enumerate(constraint_colors):
        batches[color].append(c_idx)
        
    # GPU로 보내기 좋게 numpy array로 변환
    print(f"  -> Graph Coloring Result: {num_colors} colors found.")
    return [np.array(b, dtype=np.int32) for b in batches]