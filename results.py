import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_simulation_results(csv_file='results.csv'):
    # 1. 파일 읽기
    if not os.path.exists(csv_file):
        print(f"[Error] '{csv_file}' 파일을 찾을 수 없습니다.")
        return

    df = pd.read_csv(csv_file)
    
    # 2. Warm-up (Frame 0) 제외
    df_valid = df[df['frame'] > 0].copy()
    
    if df_valid.empty:
        print("[Warning] 분석할 유효한 데이터(Frame > 0)가 없습니다.")
        return

    # 3. 주요 필드 평균 계산 (DataFrame 생성)
    # 필요한 컬럼만 선택
    cols_to_mean = ['fps', 'physics_time_ms', 'collision_time_ms', 
                    'max_penetration', 'avg_penetration', 'active_collisions']
    
    mean_values = df_valid[cols_to_mean].mean()
    
    # 보기 좋게 DataFrame으로 변환
    summary_df = pd.DataFrame(mean_values, columns=['Average'])
    summary_df = summary_df.transpose() # 행/열 전환

    print("="*60)
    print(" 📊 Simulation Performance Summary (Excluding Warm-up)")
    print("="*60)
    print(summary_df.round(4).to_string())
    print("-" * 60)
    
    # 4. 시각화 (Visualization)
    plt.figure(figsize=(15, 6))
    
    # [그래프 1] FPS 변화
    plt.subplot(1, 2, 1)
    plt.plot(df['frame'], df['fps'], marker='o', color='tab:blue', label='FPS')
    plt.axhline(mean_values['fps'], color='red', linestyle='--', label=f'Avg: {mean_values["fps"]:.2f}')
    plt.title('Frames Per Second (FPS)')
    plt.xlabel('Frame')
    plt.ylabel('FPS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # [그래프 2] Physics vs Collision Time
    plt.subplot(1, 2, 2)
    plt.plot(df['frame'], df['physics_time_ms'], marker='s', color='tab:orange', label='Total Physics Time')
    # Collision Time이 너무 크면 스케일 조정을 위해 보조축 사용 가능하지만, 일단 같이 그림
    plt.plot(df['frame'], df['collision_time_ms'], marker='x', color='tab:red', linestyle='--', label='Collision Time')
    
    plt.title('Computation Time Analysis (ms)')
    plt.xlabel('Frame')
    plt.ylabel('Time (ms)')
    plt.legend()
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 이미지 저장 및 출력
    save_path = 'simulation_report.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n[Info] 그래프가 '{save_path}'로 저장되었습니다.")
    plt.show()
    plt.close()

if __name__ == "__main__":
    # csv 파일 내용 생성 (테스트용, 실제 사용 시엔 파일이 있으므로 주석 처리하거나 무시됨)
    # 자네는 이미 파일이 있을 테니 이 부분은 넘어가도 되네.
    analyze_simulation_results(r'C:\Users\NCC\Desktop\NJ\개인\cloth-python\experiment_results/view_culling_v3_bench_1024\logs\view_culling_v3_bench_1024_metrics.csv')