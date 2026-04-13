# Baseline vs Temporal Coherence 분석

## 개요

이 문서는 `baseline_spatial_hashing`과 `temporal_coherence` 설정의 차이점을 분석하고, 왜 `temporal_coherence`가 active pair 숫자를 효과적으로 줄일 수 없는지 설명합니다.

## 설정 차이점

### 1. Baseline (baseline_spatial_hashing)

**설정:**
- `use_spatial_hashing = True`
- `use_temporal_coherence = False`
- `use_curvature_culling = False`
- `curvature_threshold = 0.0` (culling 없음)

**동작 방식:**
```python
# 매 프레임 모든 파티클의 곡률을 계산
compute_curvature_kernel()  # 전체 곡률 계산
compute_hash_kernel_v2()    # 해시 계산 (threshold=0.0이므로 culling 없음)
```

**특징:**
- 매 프레임 모든 파티클의 곡률을 **전체 재계산**
- 곡률 값은 항상 최신 상태
- `curvature_threshold = 0.0`이므로 **culling을 수행하지 않음**
- 모든 파티클이 해시 값을 가지며, 모든 파티클이 충돌 검사 대상

### 2. Temporal Coherence

**설정:**
- `use_spatial_hashing = True`
- `use_temporal_coherence = True`
- `use_curvature_culling = False`
- `curvature_threshold = 0.0` (culling 없음)

**동작 방식:**
```python
# Update Mask 계산 (움직임이 적은 파티클 식별)
compute_update_mask_kernel()  # 움직임 임계값 기반 마스크 생성

# 선택적 곡률 계산 (update_mask가 True인 파티클만 재계산)
compute_curvature_selective_kernel()  # 캐시 재사용 또는 재계산
compute_hash_kernel_v2()              # 해시 계산 (threshold=0.0이므로 culling 없음)
```

**특징:**
- **선택적 곡률 계산**: `update_mask`가 `True`인 파티클만 곡률을 재계산
- **캐시 재사용**: 움직임이 적은 파티클은 이전 프레임의 곡률 값을 재사용
- `curvature_threshold = 0.0`이므로 **culling을 수행하지 않음**
- 모든 파티클이 해시 값을 가지며, 모든 파티클이 충돌 검사 대상

## 핵심 차이점

### 1. 곡률 계산 방식

| 항목 | Baseline | Temporal Coherence |
|------|----------|-------------------|
| 계산 방식 | 전체 재계산 | 선택적 재계산 (캐시 사용) |
| 커널 | `compute_curvature_kernel` | `compute_curvature_selective_kernel` |
| 곡률 값 정확도 | 항상 최신 | 캐시된 값 사용 시 약간 오래됨 |
| 계산 비용 | 높음 (모든 파티클) | 낮음 (일부 파티클만) |

### 2. Update Mask 로직

`temporal_coherence`는 다음 조건 중 하나라도 만족하면 곡률을 재계산합니다:

```python
needs_update = (
    self_motion > motion_threshold or           # 자기 자신의 움직임이 임계값 초과
    max_neighbor_motion > motion_threshold or  # 이웃의 최대 움직임이 임계값 초과
    cache_age[idx] >= max_cache_age            # 캐시 나이가 최대값 이상
)
```

**기본 파라미터:**
- `motion_threshold = spacing * 0.05` (격자 간격의 5%)
- `max_cache_age = 5` (최대 5 substep 동안 캐시 유지)

### 3. 캐시 재사용 메커니즘

```python
# compute_curvature_selective_kernel 내부
if update_mask[idx]:
    # 곡률 재계산
    curvature_out[idx] = 새로 계산된 곡률
    curvature_cache[idx] = 새로 계산된 곡률  # 캐시 갱신
    cache_age[idx] = 0  # 나이 리셋
else:
    # 캐시 재사용
    curvature_out[idx] = curvature_cache[idx]  # 이전 값 재사용
    cache_age[idx] += 1  # 나이 증가
```

## 왜 Active Pair를 효과적으로 줄일 수 없는가?

### 핵심 문제: Culling이 수행되지 않음

**두 설정 모두 `curvature_threshold = 0.0`이므로 culling을 수행하지 않습니다.**

```python
# compute_hash_kernel_v2 내부
threshold_margin = curvature_threshold * 0.05  # 0.0 * 0.05 = 0.0
effective_threshold = curvature_threshold - threshold_margin  # 0.0 - 0.0 = 0.0

if curvature[idx] < effective_threshold:  # curvature[idx] < 0.0
    hashes[idx] = -1  # Culling
    return
```

**문제점:**
- 곡률 값은 항상 `>= 0.0` (거리 기반 계산이므로)
- `effective_threshold = 0.0`이므로 `curvature[idx] < 0.0` 조건은 **절대 참이 될 수 없음**
- 따라서 **모든 파티클이 해시 값을 가지며, 모든 파티클이 충돌 검사 대상**

### Active Pair 계산 위치

```python
# solve_self_collision_friction_kernel 내부
# 1. Hash-based Culling 체크
if particle_hashes[idx] < 0:  # Culled particle
    return

# 2. View-Dependent Culling 체크
if vis_score < view_culling_threshold:
    return

# 3. Physics-based Early Termination 체크
if dot_v_d > 1e-9:  # 두 파티클이 멀어지는 중
    continue

# 4. Active Pair 카운트 (위의 모든 체크를 통과한 pair만 카운트)
cuda.atomic.add(active_pair_count, 0, 1)
```

**결과:**
- Hash-based culling이 작동하지 않으므로 (모든 파티클이 `hash >= 0`)
- View-Dependent culling과 Physics-based Early Termination만 작동
- 따라서 **충돌 검사 대상 파티클 수는 거의 동일**

### Temporal Coherence의 한계

**Temporal Coherence는 곡률 계산 비용만 줄일 뿐, 충돌 검사 대상 수를 줄이지 않습니다:**

1. **곡률 계산 최적화**: 움직임이 적은 파티클의 곡률 재계산을 생략
2. **하지만 culling 없음**: `threshold = 0.0`이므로 모든 파티클이 충돌 검사 대상
3. **결과**: Active pair 수는 거의 동일 (약간의 차이는 캐시된 곡률 값의 부정확성 때문)

### Baseline vs Temporal Coherence Active Pair 차이의 원인

**왜 `temporal_coherence`가 평균적으로 더 많은 active pair를 가지는가?**

1. **캐시된 곡률 값의 부정확성**
   - `temporal_coherence`는 오래된 곡률 값을 사용할 수 있음
   - 하지만 `threshold = 0.0`이므로 culling에는 영향 없음
   - 해시 값 계산에는 영향 없음 (해시는 위치 기반)

2. **정렬 순서의 미세한 차이**
   - 곡률 값이 다르면 (비록 culling은 하지 않지만) 정렬 순서가 약간 달라질 수 있음
   - 이로 인해 충돌 검사 순서가 달라질 수 있음
   - 하지만 위치는 동일하므로 active pair 수는 거의 동일해야 함

3. **측정 오차**
   - GPU의 비결정적 실행 순서
   - Atomic 연산의 타이밍 차이
   - 이러한 차이는 통계적으로 유의하지 않을 수 있음

## 결론

### Temporal Coherence의 역할

**Temporal Coherence는 성능 최적화 기법이지, culling 기법이 아닙니다:**

- ✅ **골률 계산 비용 절감**: 움직임이 적은 파티클의 곡률 재계산 생략
- ❌ **Active pair 수 감소**: `curvature_threshold = 0.0`이므로 culling 불가능
- ❌ **충돌 검사 대상 감소**: 모든 파티클이 여전히 충돌 검사 대상

### Active Pair를 줄이려면

**Curvature Culling이 필요합니다:**

- `use_curvature_culling = True`
- `curvature_threshold > 0.0` (예: 0.15)
- 이 경우 곡률이 낮은 파티클이 `hash = -1`로 설정되어 충돌 검사에서 제외됨

### Baseline vs Temporal Coherence 비교

| 항목 | Baseline | Temporal Coherence |
|------|----------|-------------------|
| 곡률 계산 비용 | 높음 (전체) | 낮음 (선택적) |
| Active Pair 수 | 기준값 | 거의 동일 (약간 높을 수 있음) |
| Culling 효과 | 없음 | 없음 |
| 성능 향상 | 없음 | 곡률 계산 비용만 절감 |

**결론:** `temporal_coherence`는 곡률 계산 비용을 줄이지만, `curvature_threshold = 0.0`이므로 active pair 수를 줄이지 못합니다. Active pair를 줄이려면 `curvature_culling`을 활성화해야 합니다.
