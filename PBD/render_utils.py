import pyvista as pv
import numpy as np
import os

class ClothRenderer:
    def __init__(self, width, height, save_dir="render_output"):
        self.width = width
        self.height = height
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Topology 캐싱
        self.faces = self._create_grid_topology(width, height)
        
        # Plotter 초기화
        self.plotter = pv.Plotter(off_screen=True, window_size=(1600, 1600))
        
        # [중요] 조명 설정 보강 (입체감을 위해)
        self.plotter.enable_eye_dome_lighting()  # SSAO와 유사한 효과
        self.plotter.add_light(pv.Light(position=(0, 10, 10), intensity=0.8)) # 상단 조명

        # 카메라가 초기화되었는지 확인하는 플래그
        self.camera_set = False

    def _create_grid_topology(self, w, h):
        faces = []
        for y in range(h - 1):
            for x in range(w - 1):
                idx = y * w + x
                # Triangle 1 & 2
                faces.append([3, idx, idx + 1, idx + w + 1])
                faces.append([3, idx, idx + w + 1, idx + w])
        return np.hstack(faces)

    def render_frame(self, positions, scalar_data=None, frame_idx=0, mode="analysis"):
        """
        mode: 
          - "analysis": 히트맵(Scalar) 시각화 (기존 방식)
          - "visual": 실제 천 같은 단색 렌더링
        """
        self.plotter.clear()
        
        # 1. Mesh 생성
        mesh = pv.PolyData(positions, self.faces)
        
        # [핵심] 카메라 자동 세팅 (첫 프레임 기준)
        if not self.camera_set:
            # 메쉬의 정중앙(center)과 크기(length)를 구함
            center = mesh.center
            length = mesh.length # 대각선 길이
            
            # 카메라 위치: 중앙에서 약간 위쪽, 앞쪽으로 떨어트림
            # 숫자는 '배율'이므로 해상도와 무관하게 작동함
            cam_pos = (center[0], center[1] + length * 0.6, center[2] + length * 1.5)
            
            self.plotter.camera_position = [
                cam_pos,    # Camera Position
                center,     # Focal Point (메쉬 중앙을 바라봄)
                (0, 1, 0)   # Up Vector
            ]
            self.camera_set = True  # 이후 프레임부터는 카메라 고정 (흔들림 방지)

        # 2. 모드에 따른 렌더링 설정
        if mode == "analysis" and scalar_data is not None:
            # 분석 모드: 히트맵 사용
            mesh.point_data["values"] = scalar_data
            self.plotter.add_mesh(
                mesh, 
                scalars="values", 
                cmap="jet", 
                clim=[0, 0.05], 
                show_edges=False,
                smooth_shading=True,
                specular=0.1  # 약간의 반사광
            )
        else:
            # 비주얼 모드: 단색 천 (예: 아이보리 색)
            self.plotter.add_mesh(
                mesh, 
                color="gainsboro",  # or 'light_blue', 'white'
                show_edges=False,   # 와이어프레임 끄기
                smooth_shading=True,
                specular=0.3,       # 실크 느낌을 위한 스펙큘러
                specular_power=15
            )

        # 3. 텍스트 및 저장
        # self.plotter.add_text(f"Mode: {mode} | Frame {frame_idx:04d}", font_size=10, color="black")
        
        filename = os.path.join(self.save_dir, f"frame_{frame_idx:04d}.png")
        self.plotter.screenshot(filename)