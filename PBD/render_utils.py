import pyvista as pv
import numpy as np
import os

class ClothRenderer:
    def __init__(self, width, height, save_dir="render_output"):
        self.width = width
        self.height = height
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 1. Topology 캐싱 (Cloth)
        self.faces = self._create_grid_topology(width, height)
        
        # 2. 바닥(Floor) 미리 생성
        # 중심(0,0,0), Y축(0,1,0)을 바라보는 평면
        self.floor_mesh = pv.Plane(center=(0, -0.001, 0), direction=(0, 1, 0), i_size=250, j_size=250)
        
        # [핵심 수정 1] 기존 데이터 초기화 (이걸 안 하면 빨강/파랑 히트맵이 나옴)
        self.floor_mesh.clear_data() 
        
        # 텍스처 매핑을 위해 UV 좌표 생성
        self.floor_mesh.texture_map_to_plane(inplace=True) 
        
        # 3. 체크보드 텍스처 생성 (회색/흰색)
        self.floor_texture = self._create_checkerboard_texture()

        # 4. Plotter 초기화
        self.plotter = pv.Plotter(off_screen=True, window_size=(1600, 1600))
        
        # 조명 설정
        self.plotter.enable_eye_dome_lighting() 
        self.plotter.add_light(pv.Light(position=(0, 10, 10), intensity=0.8)) 

        # 카메라 플래그
        self.camera_set = False

    def _create_grid_topology(self, w, h):
        faces = []
        for y in range(h - 1):
            for x in range(w - 1):
                idx = y * w + x
                faces.append([3, idx, idx + 1, idx + w + 1])
                faces.append([3, idx, idx + w + 1, idx + w])
        return np.hstack(faces)

    def _create_checkerboard_texture(self):
        """
        [핵심 수정 2] float(0.0~1.0) 대신 uint8(0~255)을 사용하여
        PyVista가 색상 데이터임을 명확히 인식하게 함 (흰색/회색)
        """
        pattern_size = 32
        
        # 색상 정의 (0~255 정수 사용)
        # 흰색
        color1 = np.array([255, 255, 255], dtype=np.uint8) 
        # 회색 (밝은 회색)
        color2 = np.array([180, 180, 180], dtype=np.uint8) 
        
        # 체크 패턴 마스크 생성
        check = np.indices((pattern_size, pattern_size)).sum(axis=0) % 2
        
        # 텍스처 데이터 배열 생성 (uint8 타입)
        texture_data = np.zeros((pattern_size, pattern_size, 3), dtype=np.uint8)
        texture_data[check == 0] = color1
        texture_data[check == 1] = color2
        
        # interpolate=False: 픽셀이 뭉개지지 않고 선명한 체크무늬 유지
        return pv.Texture(texture_data, interpolate=False)

    def render_frame(self, positions, scalar_data=None, frame_idx=0, mode="analysis", sphere_params=None):
        """
        mode: 
          - "analysis": 히트맵 시각화
          - "visual": 단색 렌더링
        """
        self.plotter.clear()
        
        # ---------------------------------------------------------
        # 1. 바닥(Floor) 렌더링 [NEW]
        # ---------------------------------------------------------
        self.plotter.add_mesh(
            self.floor_mesh, 
            texture=self.floor_texture, # 체크보드 텍스처 적용
            show_edges=False,
            scalars=None,
            lighting=True,
            specular=0.1 # 약간의 반사광
        )

        # ---------------------------------------------------------
        # 2. 구체(Sphere) 렌더링
        # ---------------------------------------------------------
        if sphere_params is not None:
            center = sphere_params[:3]
            radius = sphere_params[3]
            sphere_mesh = pv.Sphere(radius=radius, center=center, phi_resolution=60, theta_resolution=60)
            
            self.plotter.add_mesh(
                sphere_mesh,
                color="orange",
                opacity=1.0,
                smooth_shading=True,
                specular=0.5,
                show_edges=False
            )

        # ---------------------------------------------------------
        # 3. 천(Cloth) 렌더링
        # ---------------------------------------------------------
        mesh = pv.PolyData(positions, self.faces)
        
        if mode == "analysis" and scalar_data is not None:
            mesh.point_data["values"] = scalar_data
            self.plotter.add_mesh(
                mesh, 
                scalars="values", 
                cmap="jet", 
                clim=[0, 0.05], 
                show_edges=False,
                smooth_shading=True,
                specular=0.25
            )
        else:
            self.plotter.add_mesh(
                mesh, 
                color="gainsboro",  
                show_edges=False,   
                smooth_shading=True,
                specular=0.3,       
                specular_power=15
            )

        # ---------------------------------------------------------
        # 4. 카메라 및 저장
        # ---------------------------------------------------------
        if not self.camera_set:
            center = mesh.center
            target = center
            if sphere_params is not None:
                target = sphere_params[:3]

            # 뷰 각도를 조금 낮춰서 더 웅장하게 (Low Angle)
            cam_pos = (target[0] + 0.0, target[1] + 30, target[2] + 70.0)
            
            self.plotter.camera_position = [cam_pos, target, (0, 1, 0)]
            self.plotter.camera.zoom(1.3) 
            self.camera_set = True

        filename = os.path.join(self.save_dir, f"frame_{frame_idx:04d}.png")
        self.plotter.screenshot(filename)