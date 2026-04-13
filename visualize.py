"""
벤치마크 결과 시각화 스크립트
CSV 파일을 읽어서 다양한 차트를 생성합니다.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def visualize_results(target_path):
    """
    벤치마크 결과 시각화
    
    Args:
        target_path: 시각화할 CSV 파일 경로
    """
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 데이터 로드
    if not os.path.exists(target_path):
        print(f"[ERROR] CSV file not found: {target_path}")
        sys.exit(1)
    
    df = pd.read_csv(target_path)
    
    # 결과 저장 디렉토리
    output_dir = os.path.dirname(target_path)
    base_name = os.path.splitext(os.path.basename(target_path))[0]
    
    sizes = sorted(df['size'].unique())
    configs = df['config'].unique()
    
    # 1. 사이즈별 평균 FPS 비교 (Culling 설정별)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(sizes))
    width = 0.2
    
    for i, config in enumerate(configs):
        config_data = df[df['config'] == config]
        avg_fps = []
        for size in sizes:
            size_data = config_data[config_data['size'] == size]
            avg_time = size_data['frame_time_ms'].mean()
            fps = 1000.0 / avg_time if avg_time > 0 else 0
            avg_fps.append(fps)
        
        ax.bar(x + i * width, avg_fps, width, label=config)
    
    ax.set_xlabel('Cloth Size')
    ax.set_ylabel('Average FPS')
    ax.set_title('Average FPS by Cloth Size and Culling Configuration')
    ax.set_xticks(x + width * (len(configs) - 1) / 2)
    ax.set_xticklabels([f'{s}x{s}' for s in sizes])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_name}_fps_comparison.png'), dpi=150)
    plt.close()
    print(f"  - FPS Comparison: {base_name}_fps_comparison.png")
    
    # 2. 프레임별 성능 추이 (사이즈별) - 각 사이즈별로 개별 차트 생성
    for size in sizes:
        size_data = df[df['size'] == size]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for config in configs:
            config_data = size_data[size_data['config'] == config]
            if len(config_data) > 0:
                # 프레임 번호로 정렬 (중요: 정렬하지 않으면 마지막과 첫 데이터가 연결됨)
                config_data = config_data.sort_values('frame')
                # 이동 평균으로 스무딩
                window = min(50, len(config_data) // 10)
                if window > 1:
                    smoothed = config_data['frame_time_ms'].rolling(window=window).mean()
                    ax.plot(config_data['frame'], smoothed, label=config, alpha=0.8, linewidth=1.5)
                else:
                    ax.plot(config_data['frame'], config_data['frame_time_ms'], label=config, alpha=0.8, linewidth=1.5)
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Frame Time (ms)')
        ax.set_title(f'Frame Time Trend: {size}x{size}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_name}_frame_time_trend_{size}x{size}.png'), dpi=150)
        plt.close()
        print(f"  - Frame Time Trend ({size}x{size}): {base_name}_frame_time_trend_{size}x{size}.png")
    
    # 3. 최적화 효과 비교 (Speedup)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_config = 'baseline_spatial_hashing'
    
    for config in configs:
        if config == baseline_config:
            continue
            
        speedups = []
        for size in sizes:
            baseline_data = df[(df['config'] == baseline_config) & (df['size'] == size)]
            config_data = df[(df['config'] == config) & (df['size'] == size)]
            
            if len(baseline_data) > 0 and len(config_data) > 0:
                baseline_time = baseline_data['frame_time_ms'].mean()
                config_time = config_data['frame_time_ms'].mean()
                speedup = baseline_time / config_time if config_time > 0 else 1.0
                speedups.append(speedup)
            else:
                speedups.append(1.0)
        
        ax.plot(sizes, speedups, 'o-', label=config, markersize=8)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1x)')
    ax.set_xlabel('Cloth Size')
    ax.set_ylabel('Speedup (vs Baseline)')
    ax.set_title('Optimization Speedup by Cloth Size')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f'{s}x{s}' for s in sizes])
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_name}_speedup.png'), dpi=150)
    plt.close()
    print(f"  - Speedup Chart: {base_name}_speedup.png")
    
    # 4. 침투 깊이 분석
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 4-1. 평균 최대 침투 깊이
    ax = axes[0]
    for config in configs:
        config_data = df[df['config'] == config]
        avg_max_pen = []
        for size in sizes:
            size_data = config_data[config_data['size'] == size]
            avg_max_pen.append(size_data['max_penetration'].mean() * 100)  # cm 단위
        ax.plot(sizes, avg_max_pen, 'o-', label=config, markersize=8)
    
    ax.set_xlabel('Cloth Size')
    ax.set_ylabel('Avg Max Penetration (cm)')
    ax.set_title('Average Maximum Penetration Depth')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f'{s}x{s}' for s in sizes])
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4-2. 활성 충돌 수
    ax = axes[1]
    for config in configs:
        config_data = df[df['config'] == config]
        avg_collisions = []
        for size in sizes:
            size_data = config_data[config_data['size'] == size]
            avg_collisions.append(size_data['active_collisions'].mean())
        ax.plot(sizes, avg_collisions, 'o-', label=config, markersize=8)
    
    ax.set_xlabel('Cloth Size')
    ax.set_ylabel('Avg Active Collisions')
    ax.set_title('Average Active Collision Count')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f'{s}x{s}' for s in sizes])
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_name}_collision_analysis.png'), dpi=150)
    plt.close()
    print(f"  - Collision Analysis: {base_name}_collision_analysis.png")
    
    # 4-3. Active Pair 변화 추이 (프레임별)
    if 'active_pairs' in df.columns:
        # 데이터 타입 변환 (문자열일 수 있음)
        df['active_pairs'] = pd.to_numeric(df['active_pairs'], errors='coerce').fillna(0)
        
        fig, axes = plt.subplots(len(sizes), 1, figsize=(14, 5 * len(sizes)))
        if len(sizes) == 1:
            axes = [axes]
        
        for idx, size in enumerate(sizes):
            ax = axes[idx]
            has_data = False
            for config in configs:
                config_data = df[(df['config'] == config) & (df['size'] == size)]
                if len(config_data) > 0:
                    # 프레임 번호로 정렬 (중요: 정렬하지 않으면 마지막과 첫 데이터가 연결됨)
                    config_data = config_data.sort_values('frame')
                    # 프레임별 active pairs 추이
                    frames = config_data['frame'].values
                    active_pairs = config_data['active_pairs'].values
                    # 0이나 음수 값 필터링 (로그 스케일 사용 시)
                    valid_mask = active_pairs > 0
                    if np.any(valid_mask):
                        ax.plot(frames[valid_mask], active_pairs[valid_mask], label=config, alpha=0.7, linewidth=1.5)
                        has_data = True
            
            if has_data:
                ax.set_xlabel('Frame')
                ax.set_ylabel('Active Pairs')
                ax.set_title(f'Active Pair Count Over Time ({size}x{size})')
                ax.legend()
                ax.grid(alpha=0.3)
                # 로그 스케일 사용 시 최소값 설정
                ax.set_yscale('log')
                ax.set_ylim(bottom=1)  # 최소값을 1로 설정하여 로그 스케일 오류 방지
            else:
                # 데이터가 없을 때 빈 차트
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Active Pair Count Over Time ({size}x{size})')
        
        try:
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{base_name}_active_pairs_trend.png'), dpi=150)
        except Exception as e:
            print(f"Warning: Could not save active_pairs_trend chart: {e}")
        finally:
            plt.close()
        
        print(f"  - Active Pairs Trend: {base_name}_active_pairs_trend.png")
        
        # 4-4. Active Pair 평균 비교
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        has_data = False
        for config in configs:
            config_data = df[df['config'] == config]
            avg_pairs = []
            valid_sizes = []
            for size in sizes:
                size_data = config_data[config_data['size'] == size]
                if 'active_pairs' in size_data.columns and len(size_data) > 0:
                    avg_val = size_data['active_pairs'].mean()
                    if avg_val > 0:  # 0보다 큰 값만 사용
                        avg_pairs.append(avg_val)
                        valid_sizes.append(size)
                        has_data = True
            
            if len(avg_pairs) > 0:
                ax.plot(valid_sizes, avg_pairs, 'o-', label=config, markersize=8)
        
        if has_data:
            ax.set_xlabel('Cloth Size')
            ax.set_ylabel('Avg Active Pairs')
            ax.set_title('Average Active Pair Count by Configuration')
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_ylim(bottom=1)  # 최소값을 1로 설정
            ax.set_xticks(sizes)
            ax.set_xticklabels([f'{s}x{s}' for s in sizes])
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Average Active Pair Count by Configuration')
        
        try:
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{base_name}_active_pairs_comparison.png'), dpi=150)
        except Exception as e:
            print(f"Warning: Could not save active_pairs_comparison chart: {e}")
        finally:
            plt.close()
        
        print(f"  - Active Pairs Comparison: {base_name}_active_pairs_comparison.png")
    
    # 5. 요약 통계 테이블 생성 (mean±std 지원)
    summary_data = []
    
    # trial 컬럼이 있는지 확인 (다중 시행 여부)
    has_trials = 'trial' in df.columns
    
    for size in sizes:
        for config in configs:
            data = df[(df['size'] == size) & (df['config'] == config)]
            if len(data) > 0:
                if has_trials:
                    # 다중 시행: trial별 평균 FPS 계산 후 mean±std
                    trials = data['trial'].unique()
                    trial_fps_list = []
                    trial_time_list = []
                    
                    for trial in trials:
                        trial_data = data[data['trial'] == trial]
                        avg_time = trial_data['frame_time_ms'].mean()
                        fps = 1000.0 / avg_time if avg_time > 0 else 0
                        trial_fps_list.append(fps)
                        trial_time_list.append(avg_time)
                    
                    fps_mean = np.mean(trial_fps_list)
                    fps_std = np.std(trial_fps_list)
                    time_mean = np.mean(trial_time_list)
                    time_std = np.std(trial_time_list)
                    
                    avg_pairs = data['active_pairs'].mean() if 'active_pairs' in data.columns else 0
                    summary_data.append({
                        'Size': f'{size}x{size}',
                        'Config': config,
                        'Num Trials': len(trials),
                        'FPS (mean)': round(fps_mean, 2),
                        'FPS (std)': round(fps_std, 2),
                        'FPS (mean±std)': f'{fps_mean:.2f}±{fps_std:.2f}',
                        'Frame Time (mean ms)': round(time_mean, 3),
                        'Frame Time (std ms)': round(time_std, 3),
                        'Avg Max Pen (cm)': round(data['max_penetration'].mean() * 100, 4),
                        'Avg Collisions': round(data['active_collisions'].mean(), 1),
                        'Avg Active Pairs': round(avg_pairs, 0)
                    })
                else:
                    # 단일 시행
                    avg_time = data['frame_time_ms'].mean()
                    fps = 1000.0 / avg_time if avg_time > 0 else 0
                    avg_pairs = data['active_pairs'].mean() if 'active_pairs' in data.columns else 0
                    summary_data.append({
                        'Size': f'{size}x{size}',
                        'Config': config,
                        'Num Trials': 1,
                        'FPS (mean)': round(fps, 2),
                        'FPS (std)': 0.0,
                        'FPS (mean±std)': f'{fps:.2f}±0.00',
                        'Frame Time (mean ms)': round(avg_time, 3),
                        'Frame Time (std ms)': 0.0,
                        'Avg Max Pen (cm)': round(data['max_penetration'].mean() * 100, 4),
                        'Avg Collisions': round(data['active_collisions'].mean(), 1),
                        'Avg Active Pairs': round(avg_pairs, 0)
                    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f'{base_name}_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"  - Summary Table: {base_name}_summary.csv")
    
    # 6. mean±std 막대 그래프 (에러바 포함)
    if has_trials:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(sizes))
        width = 0.2
        
        for i, config in enumerate(configs):
            config_summary = summary_df[summary_df['Config'] == config]
            fps_means = config_summary['FPS (mean)'].values
            fps_stds = config_summary['FPS (std)'].values
            
            ax.bar(x + i * width, fps_means, width, label=config, yerr=fps_stds, capsize=3)
        
        ax.set_xlabel('Cloth Size')
        ax.set_ylabel('FPS (mean ± std)')
        ax.set_title(f'Performance Comparison with Error Bars ({len(df["trial"].unique())} trials)')
        ax.set_xticks(x + width * (len(configs) - 1) / 2)
        ax.set_xticklabels([f'{s}x{s}' for s in sizes])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_name}_fps_with_errorbar.png'), dpi=150)
        plt.close()
        
        print(f"  - FPS with Error Bars: {base_name}_fps_with_errorbar.png")
    
    print(f"\n[CHART] Visualization Complete!")
    print(f"Output directory: {output_dir}")
    
    return summary_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Benchmark Results')
    parser.add_argument('target_path', type=str, help='Path to CSV file to visualize')
    
    args = parser.parse_args()
    
    visualize_results(args.target_path)
