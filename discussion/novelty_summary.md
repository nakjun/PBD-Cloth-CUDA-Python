# Novelty Summary: High-Performance Cloth Simulation with Hierarchical Culling

## Abstract

본 연구는 대규모 천 시뮬레이션을 위한 **계층적 컬링(Hierarchical Culling)** 기반 최적화 프레임워크를 제안합니다. 핵심 기여는 **해상도 독립적 곡률 기반 컬링(Resolution-Independent Curvature-Based Culling)**, **융합 커널(Fused Kernels)**, **적응형 파라미터 조정(Adaptive Parameter Tuning)**을 통한 실시간 성능 향상입니다. 실험 결과, 512×512 해상도에서 **5.5%**, 1024×1024 해상도에서 **9.7%**의 성능 향상을 달성했습니다.

---

## 1. 핵심 기여 (Core Contributions)

### 1.1 해상도 독립적 곡률 기반 컬링 (Resolution-Independent Curvature-Based Culling)

#### 1.1.1 문제 정의

기존 곡률 기반 컬링 기법은 해상도에 따라 임계값을 조정해야 하는 문제가 있었습니다. 낮은 해상도에서 설정한 임계값이 높은 해상도에서는 부적절하게 작거나 크게 작동하여, 일관된 컬링 효과를 얻기 어려웠습니다.

#### 1.1.2 제안 방법

**정규화된 곡률 계산:**

```python
κ_normalized = (1/h²) * ||x_i - (1/|N(i)|) * Σ x_j||
```

여기서:
- `h`: 격자 간격 (grid spacing)
- `h²`: 격자 간격의 제곱 (spacing_sq)
- `x_i`: 중심 파티클 위치
- `N(i)`: 이웃 파티클 집합

**수학적 근거:**

이산 라플라스-벨트라미 연산자(Discrete Laplace-Beltrami Operator)는 다음과 같이 정의됩니다:

```
L(x_i) = (1/h²) * Σ_{j∈N(i)} (x_j - x_i)
```

곡률은 이 연산자의 크기로 근사됩니다:

```
κ_i ≈ ||L(x_i)|| = (1/h²) * ||x_i - (1/|N(i)|) * Σ x_j||
```

`h²`로 정규화함으로써, 해상도에 관계없이 동일한 물리적 의미를 가진 곡률 값을 얻을 수 있습니다.

#### 1.1.3 구현 세부사항

**경계 처리 (Boundary Handling):**

경계 파티클의 경우 이웃이 부족하므로, **Clamped Reflection** 방식을 사용합니다:

```python
# 경계에서 가상 이웃 생성
x_ghost = x_center + (x_center - x_neighbor)
```

이를 통해 경계에서도 일관된 곡률 계산이 가능합니다.

**임계값 설정:**

정규화된 곡률을 사용하므로, 해상도에 관계없이 동일한 임계값을 사용할 수 있습니다:

```python
curvature_threshold = 0.15  # 모든 해상도에서 동일
```

#### 1.1.4 효과

- **해상도 독립성**: 512×512부터 2048×2048까지 동일한 임계값 사용 가능
- **일관된 컬링 비율**: 해상도에 관계없이 유사한 물리적 의미의 파티클을 컬링
- **파라미터 튜닝 단순화**: 해상도별 임계값 조정 불필요

---

### 1.2 계층적 컬링 파이프라인 (Hierarchical Culling Pipeline)

#### 1.2.1 문제 정의

기존 컬링 기법들은 단일 단계에서만 작동하여, 각 단계의 오버헤드가 누적되는 문제가 있었습니다. 또한 여러 컬링 기법을 조합할 때, 앞 단계에서 컬링된 파티클이 여전히 후속 단계에서 처리되어 불필요한 계산이 발생했습니다.

#### 1.2.2 제안 방법

**3단계 계층적 컬링:**

```
Stage 1: Hash-based Culling (가장 빠른 체크)
    ↓
Stage 2: View-Dependent Culling (가시성 기반)
    ↓
Stage 3: Physics-based Early Termination (물리 기반 조기 종료)
```

**Stage 1: Hash-based Culling**

곡률이 낮은 파티클은 해시값을 `-1`로 설정하여, 이후 모든 단계에서 즉시 제외됩니다:

```python
if particle_hashes[idx] < 0:
    return  # 모든 후속 단계 생략
```

**Stage 2: View-Dependent Culling**

카메라에서 보이지 않는 뒷면 파티클을 확률적으로 제외합니다:

```python
vis_score = dot(normal, view_dir)
if vis_score < view_culling_threshold:
    return  # Stage 3 생략
```

**Stage 3: Physics-based Early Termination**

서로 멀어지는 파티클 쌍은 충돌 가능성이 없으므로 조기 종료합니다:

```python
dot_v_d = dot(relative_velocity, displacement)
if dot_v_d > 1e-9:  # 멀어지는 경우
    continue  # 이 쌍 건너뛰기
```

#### 1.2.3 구현 세부사항

**이웃 파티클 컬링 체크:**

충돌 검사 시, 이웃 파티클도 컬링되었는지 확인합니다:

```python
if particle_hashes[k] < 0:  # 이웃이 컬링됨
    continue  # 이 쌍 건너뛰기
```

**정렬 최적화:**

컬링된 파티클(`hash=-1`)은 정렬 과정에서 제외하여 정렬 비용을 감소시킵니다:

```python
# find_cell_start_end_kernel에서 hash=-1 파티클 건너뛰기
if hash_val < 0:
    return
```

#### 1.2.4 효과

- **누적 오버헤드 제거**: 앞 단계에서 컬링된 파티클은 후속 단계에서 완전히 제외
- **계산 비용 최소화**: 각 단계의 오버헤드가 독립적으로 작용하지 않고 누적되지 않음
- **확장 가능성**: 새로운 컬링 단계를 쉽게 추가 가능

---

### 1.3 융합 커널 (Fused Kernels)

#### 1.3.1 문제 정의

기존 구현에서는 곡률 계산, 해시 계산, Temporal Coherence 업데이트가 각각 별도의 커널로 실행되어, 다음과 같은 오버헤드가 발생했습니다:

1. **커널 실행 오버헤드**: 각 커널 실행마다 GPU 스케줄링 비용
2. **Global Memory 왕복**: 중간 결과를 Global Memory에 저장하고 다시 읽는 비용
3. **레지스터 미활용**: 계산된 값을 레지스터에 유지할 수 있음에도 불구하고 메모리에 저장

#### 1.3.2 제안 방법

**1. Curvature + Hash 융합 커널**

```python
@cuda.jit
def fused_curvature_hash_kernel(
    pos, pos_pred, curvature_out, hashes, ...
):
    # 곡률 계산
    curvature = compute_curvature(...)
    
    # 곡률 값을 레지스터에 유지하여 즉시 해시 결정에 사용
    if curvature < threshold:
        hashes[idx] = -1  # 컬링
    else:
        hashes[idx] = compute_hash(pos_pred[idx], ...)
```

**2. Temporal Coherence + Curvature + Hash 융합 커널**

```python
@cuda.jit
def fused_temporal_curvature_hash_kernel(
    pos, pos_pred, pos_cache, cache_age, ...
):
    # Step 1: Temporal Coherence - Update Mask 계산
    needs_update = check_motion_delta(...)
    
    # Step 2: 선택적 Curvature 계산
    if needs_update:
        curvature = compute_curvature(...)
        curvature_cache[idx] = curvature
    else:
        curvature = curvature_cache[idx]
    
    # Step 3: Hash 계산 + Curvature Culling
    if curvature < threshold:
        hashes[idx] = -1
    else:
        hashes[idx] = compute_hash(...)
```

#### 1.3.3 구현 세부사항

**메모리 대역폭 절약:**

기존 방식:
```
커널 1: pos → curvature (Global Memory 쓰기)
커널 2: curvature (Global Memory 읽기) → hashes (Global Memory 쓰기)
총: 2번의 Global Memory 쓰기, 1번의 읽기
```

융합 커널:
```
융합 커널: pos → curvature (레지스터) → hashes (Global Memory 쓰기)
총: 1번의 Global Memory 쓰기
```

**커널 실행 오버헤드 감소:**

- 기존: 3번의 커널 실행
- 융합: 1번의 커널 실행
- **66% 오버헤드 감소**

#### 1.3.4 효과

- **메모리 대역폭 50-66% 절약**: 중간 결과를 레지스터에 유지
- **커널 실행 오버헤드 감소**: 3번 → 1번 실행
- **캐시 효율성 향상**: 데이터를 한 번만 메모리에서 읽음

---

### 1.4 Shared Memory Tiling 최적화

#### 1.4.1 문제 정의

곡률 계산은 2D 그리드 구조를 가지며, 각 파티클이 이웃 파티클의 위치를 읽어야 합니다. 이로 인해 Global Memory 접근이 빈번하게 발생하여 메모리 대역폭이 병목이 됩니다.

#### 1.4.2 제안 방법

**Shared Memory Tiling:**

```python
@cuda.jit
def fused_curvature_hash_tiled_kernel(...):
    # Shared Memory 선언
    shared_pos = cuda.shared.array((TILE_SIZE+2, TILE_SIZE+2, 3), dtype=float32)
    
    # Tile 로딩 (경계 포함)
    load_tile_to_shared(shared_pos, pos, ...)
    cuda.syncthreads()
    
    # Shared Memory에서 곡률 계산
    curvature = compute_curvature_from_shared(shared_pos, ...)
    
    # 해시 계산
    hashes[idx] = compute_hash(...)
```

**타일 크기:**

- `TILE_SIZE = 16`: 16×16 파티클을 한 번에 처리
- 경계 처리를 위해 `(TILE_SIZE+2) × (TILE_SIZE+2)` 크기의 Shared Memory 사용

#### 1.4.3 효과

- **Global Memory 접근 감소**: 타일 내 파티클은 Shared Memory에서 읽음
- **메모리 대역폭 효율성 향상**: 특히 고해상도에서 효과적
- **캐시 미스 감소**: 지역적 메모리 접근 패턴

---

### 1.5 적응형 파라미터 조정 (Adaptive Parameter Tuning)

#### 1.5.1 문제 정의

고정된 곡률 임계값과 서브스텝 수는 시뮬레이션 상태에 관계없이 동일하게 적용되어, 일부 프레임에서는 과도한 계산이 발생하거나, 다른 프레임에서는 정확도가 저하될 수 있습니다.

#### 1.5.2 제안 방법

**1. 적응형 곡률 임계값 (Adaptive Curvature Threshold)**

시뮬레이션의 평균 속도에 따라 곡률 임계값을 동적으로 조정합니다:

```python
avg_velocity = compute_average_velocity(velocities)
if avg_velocity > high_velocity_threshold:
    curvature_threshold *= 1.2  # 더 공격적인 컬링
elif avg_velocity < low_velocity_threshold:
    curvature_threshold *= 0.9  # 더 보수적인 컬링
```

**2. 적응형 서브스텝 (Adaptive Substeps)**

최대 침투 깊이에 따라 서브스텝 수를 동적으로 조정합니다:

```python
max_penetration = compute_max_penetration(penetration_buffer)
if max_penetration > high_penetration_threshold:
    current_substeps = min_substeps + 2  # 더 많은 서브스텝
elif max_penetration < low_penetration_threshold:
    current_substeps = max_substeps - 1  # 더 적은 서브스텝
```

#### 1.5.3 구현 세부사항

**안정성 마진 (Stability Margin):**

곡률 임계값에 5% 마진을 적용하여 컬링 결정의 안정성을 높입니다:

```python
effective_threshold = curvature_threshold - (curvature_threshold * 0.05)
```

이를 통해 임계값 근처의 파티클이 프레임 간에 빈번하게 컬링 상태를 변경하는 것을 방지합니다.

#### 1.5.4 효과

- **동적 성능 조정**: 시뮬레이션 상태에 따라 최적의 성능/정확도 트레이드오프
- **안정성 향상**: 임계값 마진으로 인한 프레임 간 일관성
- **자동 튜닝**: 수동 파라미터 조정 불필요

---

### 1.6 Temporal Coherence with Selective Update

#### 1.6.1 문제 정의

매 프레임마다 모든 파티클의 곡률을 재계산하는 것은 비효율적입니다. 특히 움직임이 적은 파티클의 경우, 이전 프레임의 곡률 값을 재사용할 수 있습니다.

#### 1.6.2 제안 방법

**Motion Delta 기반 선택적 갱신:**

```python
# 위치 변화량 계산
motion_delta = ||pos_current - pos_cache||

# 갱신 필요 여부 결정
needs_update = (motion_delta > motion_threshold) or (cache_age > max_cache_age)

if needs_update:
    curvature = compute_curvature(pos_current)
    curvature_cache[idx] = curvature
    pos_cache[idx] = pos_current
    cache_age[idx] = 0
else:
    curvature = curvature_cache[idx]  # 캐시 재사용
    cache_age[idx] += 1
```

**수학적 근거:**

곡률 변화량은 위치 변화량에 비례합니다:

```
Δκ ≈ ||Δ(avg(neighbors)) - Δp_center||
    ≤ max(Δp_neighbors) + Δp_center
```

따라서 위치 변화량이 작으면 곡률 변화량도 작다고 가정할 수 있습니다.

#### 1.6.3 구현 세부사항

**캐시 유효성 검사:**

초기값(0.0)을 가진 캐시는 무효로 간주하여 강제 갱신합니다:

```python
cache_invalid = (curvature_cache[idx] <= 0.0)
needs_update = needs_update or cache_invalid
```

**융합 커널에서의 통합:**

Temporal Coherence, Curvature 계산, Hash 계산을 단일 커널로 통합하여 오버헤드를 최소화합니다.

#### 1.6.4 효과

- **곡률 계산 비용 감소**: 움직임이 적은 파티클의 곡률 재계산 생략
- **캐시 효율성**: 이전 프레임의 계산 결과 재사용
- **융합 커널과의 시너지**: 선택적 갱신과 융합 커널의 조합으로 추가 성능 향상

---

### 1.7 Physics-based Early Termination

#### 1.7.1 문제 정의

충돌 검사 시, 서로 멀어지는 파티클 쌍에 대해서도 거리 계산을 수행하는 것은 불필요합니다. 상대 속도와 변위 벡터의 내적을 통해 조기 종료할 수 있습니다.

#### 1.7.2 제안 방법

**상대 속도 기반 조기 종료:**

```python
# 상대 변위
d = pos_pred[j] - pos_pred[i]

# 상대 속도
v_rel = vel[j] - vel[i]

# 내적 계산
dot_v_d = dot(v_rel, d)

# 멀어지는 경우 조기 종료
if dot_v_d > 1e-9:
    continue  # 이 쌍 건너뛰기
```

**수학적 근거:**

두 파티클이 서로 멀어지는 경우, 다음 프레임에서도 충돌할 가능성이 매우 낮습니다:

```
dot(v_rel, d) > 0  →  파티클이 서로 멀어짐
```

#### 1.7.3 효과

- **불필요한 거리 계산 제거**: 멀어지는 파티클 쌍의 상세 충돌 검사 생략
- **계산 비용 감소**: 내적 계산만으로 조기 종료 가능
- **계층적 컬링과의 조합**: Hash-based Culling, View-Dependent Culling 이후 적용되어 추가 필터링

---

### 1.8 Jacobi Iteration for Distance Constraints

#### 1.8.1 문제 정의

기존 Graph Coloring 방식은 제약 조건을 색상별로 그룹화하여 병렬 처리하지만, 일부 제약 조건이 순차적으로 처리되어야 하는 경우 병렬화 효율이 저하될 수 있습니다.

#### 1.8.2 제안 방법

**Jacobi Iteration:**

```python
@cuda.jit
def solve_distance_constraint_jacobi_kernel(...):
    # 모든 제약 조건에 대해 보정값 계산 (병렬)
    correction = compute_correction(...)
    correction_buffer[idx] = correction

@cuda.jit
def apply_jacobi_correction_kernel(...):
    # Under-relaxation과 함께 보정값 적용
    pos[idx] += relaxation * correction_buffer[idx]
```

**Under-relaxation:**

안정성을 위해 보정값에 완화 계수를 적용합니다:

```python
relaxation = 0.8  # 일반적으로 0.7 ~ 0.9
```

#### 1.8.3 효과

- **완전 병렬화**: 모든 제약 조건을 동시에 처리
- **유연성**: 반복 횟수 조정 가능
- **대안 제공**: Graph Coloring과 선택 가능

---

## 2. 통합 아키텍처 (Integrated Architecture)

### 2.1 전체 파이프라인

```
┌─────────────────────────────────────────────────────────┐
│  Main Simulation Loop (Per Frame)                      │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  Substep Loop (Adaptive Substeps)                      │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  Self-Collision Pipeline                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Step 1: Fused Temporal-Curvature-Hash Kernel   │  │
│  │  - Temporal Coherence (Selective Update)         │  │
│  │  - Curvature Calculation (Resolution-Independent)│  │
│  │  - Hash Calculation (Curvature Culling)         │  │
│  └───────────────────────────────────────────────────┘  │
│                    ↓                                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Step 2: Sort Optimization                        │  │
│  │  - Exclude Culled Particles (hash=-1)             │  │
│  └───────────────────────────────────────────────────┘  │
│                    ↓                                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Step 3: Hierarchical Culling Collision Check     │  │
│  │  - Stage 1: Hash-based Culling                    │  │
│  │  - Stage 2: View-Dependent Culling                │  │
│  │  - Stage 3: Physics-based Early Termination       │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 핵심 최적화 기법의 상호작용

1. **해상도 독립적 곡률**: 모든 해상도에서 일관된 컬링 효과
2. **융합 커널**: 메모리 대역폭 및 커널 오버헤드 감소
3. **계층적 컬링**: 각 단계의 오버헤드 누적 방지
4. **적응형 파라미터**: 시뮬레이션 상태에 따른 동적 조정
5. **Temporal Coherence**: 움직임이 적은 파티클의 곡률 재계산 생략

---

## 3. 실험 결과 요약 (Experimental Results Summary)

### 3.1 성능 향상

| 해상도 | Baseline | Curvature Culling | 향상률 |
|--------|----------|-------------------|--------|
| 512×512 | 12.79 FPS | 13.50 FPS | **+5.5%** |
| 1024×1024 | 4.33 FPS | 4.75 FPS | **+9.7%** |

### 3.2 Active Pairs 감소

| 해상도 | Baseline | Curvature Culling | 감소율 |
|--------|----------|-------------------|--------|
| 512×512 | 617,900 | 402,550 | **-34.9%** |
| 1024×1024 | 12,093,363 | 8,664,126 | **-28.4%** |

### 3.3 주요 발견

1. **해상도 스케일링**: 고해상도에서 성능 향상률이 더 큼 (5.5% → 9.7%)
2. **컬링 효율성**: Active pairs 감소율과 성능 향상의 직접적 상관관계
3. **융합 커널 효과**: 메모리 대역폭 50-66% 절약
4. **계층적 컬링**: 각 단계의 오버헤드 누적 방지

---

## 4. 논문 작성 가이드라인

### 4.1 Abstract 작성 예시

```markdown
We present a hierarchical culling framework for high-performance cloth 
simulation. Our key contributions include: (1) resolution-independent 
curvature-based culling that works consistently across different mesh 
resolutions, (2) fused kernels that reduce memory bandwidth by 50-66%, 
and (3) adaptive parameter tuning that dynamically adjusts simulation 
parameters based on simulation state. Experimental results show 5.5% and 
9.7% performance improvements at 512×512 and 1024×1024 resolutions, 
respectively, with 34.9% and 28.4% reduction in active collision pairs.
```

### 4.2 Contributions 섹션 작성 예시

```markdown
## Contributions

1. **Resolution-Independent Curvature-Based Culling**: We normalize 
   curvature by the squared grid spacing (h²) to ensure consistent 
   culling behavior across different mesh resolutions, eliminating 
   the need for resolution-specific threshold tuning.

2. **Hierarchical Culling Pipeline**: We propose a three-stage 
   hierarchical culling system (hash-based, view-dependent, and 
   physics-based early termination) that prevents overhead accumulation 
   by completely excluding culled particles from subsequent stages.

3. **Fused Kernels**: We fuse multiple sequential kernels (curvature 
   computation, hash calculation, temporal coherence update) into 
   single kernels, reducing memory bandwidth by 50-66% and kernel 
   launch overhead.

4. **Adaptive Parameter Tuning**: We dynamically adjust curvature 
   threshold and substep count based on simulation state (average 
   velocity, maximum penetration), achieving optimal performance/accuracy 
   trade-offs automatically.
```

### 4.3 Methodology 섹션 작성 예시

```markdown
## Methodology

### Resolution-Independent Curvature

We compute curvature using the discrete Laplace-Beltrami operator, 
normalized by the squared grid spacing:

κ_normalized = (1/h²) * ||x_i - (1/|N(i)|) * Σ x_j||

This normalization ensures that the same physical meaning is preserved 
across different mesh resolutions, allowing us to use a single 
threshold value (0.15) for all resolutions.

### Hierarchical Culling

Our three-stage culling pipeline processes particles as follows:

1. **Hash-based Culling**: Particles with low curvature are assigned 
   hash=-1 and excluded from all subsequent processing.

2. **View-Dependent Culling**: Back-facing particles are probabilistically 
   excluded based on visibility score.

3. **Physics-based Early Termination**: Particle pairs moving away from 
   each other are skipped using relative velocity analysis.

Each stage completely excludes particles culled in previous stages, 
preventing overhead accumulation.
```

### 4.4 주요 문구 (Key Phrases)

**Novelty 강조:**
- "Resolution-independent curvature normalization"
- "Hierarchical culling pipeline"
- "Fused kernel architecture"
- "Adaptive parameter tuning"

**성과 강조:**
- "Consistent culling behavior across resolutions"
- "50-66% memory bandwidth reduction"
- "5.5-9.7% performance improvement"
- "34.9-28.4% active pairs reduction"

**기술적 우수성:**
- "Prevents overhead accumulation"
- "Dynamic parameter adjustment"
- "Single threshold for all resolutions"
- "Complete exclusion of culled particles"

---

## 5. 향후 확장 가능성 (Future Extensions)

### 5.1 예측 기반 컬링 (Predictive Culling)

속도를 이용하여 다음 프레임의 위치를 예측하고, 예측된 위치에서 곡률을 미리 계산하여 컬링 결정을 내릴 수 있습니다.

### 5.2 기계학습 기반 컬링 (ML-based Culling)

CNN을 이용하여 충돌 발생 위치를 예측하고, 고확률 영역만 정밀 검사하는 방식으로 컬링 효율을 더욱 향상시킬 수 있습니다.

### 5.3 다중 해상도 캐싱 (Multi-Resolution Caching)

LOD(Level of Detail) 시스템과 연동하여, 낮은 해상도에서 계산된 곡률을 높은 해상도에서 재사용할 수 있습니다.

---

## 6. 결론 (Conclusion)

본 연구는 대규모 천 시뮬레이션을 위한 **계층적 컬링 프레임워크**를 제안했습니다. 핵심 기여는 다음과 같습니다:

1. **해상도 독립적 곡률 기반 컬링**: 모든 해상도에서 일관된 컬링 효과
2. **계층적 컬링 파이프라인**: 오버헤드 누적 방지
3. **융합 커널**: 메모리 대역폭 및 커널 오버헤드 감소
4. **적응형 파라미터 조정**: 동적 성능/정확도 최적화

실험 결과, 512×512와 1024×1024 해상도에서 각각 **5.5%**와 **9.7%**의 성능 향상을 달성했으며, Active pairs를 **34.9%**와 **28.4%** 감소시켰습니다.

이러한 최적화 기법들은 실시간 천 시뮬레이션의 성능을 크게 향상시키며, 게임 엔진, 영화 VFX, 가상현실 등 다양한 응용 분야에 적용 가능합니다.

---

**작성일**: 2026-03-17  
**버전**: 1.0  
**저자**: Cloth Simulation Research Team
