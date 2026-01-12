import imageio.v3 as iio
import os
import glob
from pathlib import Path

def create_gif_from_folder(source_folder, output_filename="simulation_result.gif", fps=30):
    """
    지정된 폴더의 PNG 파일들을 읽어 GIF로 변환합니다.
    
    Args:
        source_folder (str): 이미지가 저장된 폴더 경로
        output_filename (str): 저장할 GIF 파일명
        fps (int): 초당 프레임 수 (속도 조절)
    """
    
    # 1. 경로 검증
    if not os.path.exists(source_folder):
        print(f"[Error] 폴더를 찾을 수 없습니다: {source_folder}")
        return

    # 2. 파일 검색 (frame_*.png 패턴)
    # glob을 사용하여 해당 패턴의 모든 파일을 찾습니다.
    search_pattern = os.path.join(source_folder, "frame_*.png")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"[Warning] '{source_folder}' 폴더에 'frame_*.png' 파일이 없습니다.")
        return

    # 3. 파일 정렬 (Natural Sorting)
    # 컴퓨터는 기본적으로 1, 10, 2 순서로 정렬하므로, 숫자를 기준으로 다시 정렬해야 합니다.
    # 가정: 파일명이 '..._숫자.png' 형식을 따른다고 가정합니다.
    try:
        files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    except ValueError:
        print("[Warning] 파일명 끝에 숫자가 없는 파일이 섞여 있어 이름순 정렬을 수행합니다.")
        files.sort()

    print(f"[Info] 총 {len(files)}개의 프레임을 발견했습니다. GIF 생성을 시작합니다...")

    # 4. 이미지 로드 및 GIF 생성
    frames = []
    # 진행 상황을 보여주기 위해 간단한 로그 출력
    for i, file_path in enumerate(files):
        if i % 50 == 0:
            print(f"  - Processing: {i}/{len(files)}")
        frames.append(iio.imread(file_path))

    # 5. 저장 (Loop=0은 무한 반복을 의미)
    # duration은 한 프레임이 보여지는 시간(ms)입니다. 1000ms / fps
    duration_ms = 1000 / fps
    
    # imageio v3 API 사용
    iio.imwrite(output_filename, frames, duration=duration_ms, loop=0)
    
    print(f"\n[Success] GIF 생성이 완료되었습니다!")
    print(f"  - 저장 위치: {os.path.abspath(output_filename)}")
    print(f"  - FPS: {fps}")

if __name__ == "__main__":
    # --- 사용 예시 ---

    input_dir = '../xpbd_wind_SDF_v6'
    output_name = '../xbpd_wind_SDF_v6.gif'

    # 함수 실행
    create_gif_from_folder(input_dir, output_name, fps=60)