import bpy
import os
import re
import math
import mathutils

# ==========================================
# [Global] 색상 데이터 캐싱용 저장소
# { frame_index: [r, g, b, a, r, g, b, a, ...] }
# ==========================================
COLOR_CACHE = {}

def clear_scene():
    """씬 초기화 및 핸들러 정리"""
    global COLOR_CACHE
    COLOR_CACHE = {}  # 캐시 초기화
    
    # 기존 객체 삭제
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # 미사용 데이터 블록 삭제 (Orphan Data)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
            
    # 기존 핸들러 제거 (중복 실행 방지)
    if update_colors_per_frame in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(update_colors_per_frame)
    
    print("🧹 씬과 핸들러가 초기화되었습니다.")

def sort_obj_files_naturally(file_list):
    """파일명을 숫자 기준으로 자연스럽게 정렬"""
    def key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', s)]
    return sorted(file_list, key=key)

def focus_camera_on_object(obj, margin=1.2):
    """카메라 자동 포커싱"""
    local_bbox_center = 0.125 * sum((mathutils.Vector(b) for b in obj.bound_box), mathutils.Vector())
    global_bbox_center = obj.matrix_world @ local_bbox_center
    
    # Bounding Box 계산
    bound_points = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    # unpacking error 방지를 위해 리스트 컴프리헨션 사용
    xs = [v.x for v in bound_points]
    ys = [v.y for v in bound_points]
    zs = [v.z for v in bound_points]
    
    max_coord = mathutils.Vector((max(xs), max(ys), max(zs)))
    min_coord = mathutils.Vector((min(xs), min(ys), min(zs)))
    
    size = max_coord - min_coord
    max_dim = max(size) * margin
    
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        bpy.ops.object.camera_add(location=(8, -8, 6))
        cam = bpy.context.object
        scene.camera = cam
        
    cam_data = cam.data
    if cam_data.type != 'PERSP':
        cam_data.type = 'PERSP'
        
    fov = cam_data.angle
    distance = (max_dim / 2) / math.tan(fov / 2)
    
    # 카메라 위치 설정 (쿼터뷰 느낌)
    cam.location = (global_bbox_center.x + distance * 0.7, 
                    global_bbox_center.y - distance * 0.7, 
                    global_bbox_center.z + distance * 0.5)
    
    # 트랙킹 제약조건 추가
    constraint = cam.constraints.get('Track To') or cam.constraints.new('TRACK_TO')
    constraint.target = obj
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    print('📷 카메라가 오브젝트에 포커싱되었습니다.')

def setup_lighting():
    """3점 조명 설정"""
    # Key Light
    bpy.ops.object.light_add(type='AREA', location=(5, -5, 8))
    key = bpy.context.object
    key.data.energy = 1000
    key.data.size = 10
    
    # Fill Light
    bpy.ops.object.light_add(type='AREA', location=(-5, -3, 5))
    fill = bpy.context.object
    fill.data.energy = 500
    fill.data.color = (0.9, 0.95, 1.0)
    
    # Rim Light
    bpy.ops.object.light_add(type='SPOT', location=(0, 5, 6))
    rim = bpy.context.object
    rim.data.energy = 800
    rim.rotation_euler = (1.57, 0, 3.14)
    
    print("💡 조명 설정 완료.")

def animate_shape_keys(obj, total_frames):
    """Shape Key 애니메이션 키프레임 등록"""
    shape_keys = obj.data.shape_keys.key_blocks
    
    # Basis는 항상 1.0 유지 (혹은 필요에 따라 조절)
    shape_keys[0].value = 1.0
    
    # 프레임별로 Shape Key 켜고 끄기
    for i, sk in enumerate(shape_keys[1:]):  # Basis 제외
        fn = i + 1 # 실제 프레임 번호 (1부터 시작한다고 가정)
        
        # 이전 프레임: 0.0
        if fn > 1:
            sk.value = 0.0
            sk.keyframe_insert(data_path="value", frame=fn - 1)
        
        # 현재 프레임: 1.0
        sk.value = 1.0
        sk.keyframe_insert(data_path="value", frame=fn)
        
        # 다음 프레임: 0.0
        if fn < total_frames:
            sk.value = 0.0
            sk.keyframe_insert(data_path="value", frame=fn + 1)

def setup_heatmap_material(obj):
    """Vertex Color(Attribute)를 시각화하는 재질 생성"""
    mat_name = "HeatmapMaterial"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Nodes 생성
    output = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    attribute = nodes.new(type='ShaderNodeAttribute')
    
    # [핵심] Blender Importer가 생성한 속성 이름 지정 (보통 'Color' 아니면 'Col')
    # 아래 import 로직에서 확인된 이름을 사용
    target_attr_name = "Color"
    if "Col" in obj.data.attributes:
        target_attr_name = "Col"
    
    attribute.attribute_name = target_attr_name
    
    # 연결: Attribute Color -> Base Color & Emission (잘 보이게)
    links.new(attribute.outputs['Color'], bsdf.inputs['Base Color'])
    
    # 살짝 빛나게 해서 빨간색(충돌) 강조
    links.new(attribute.outputs['Color'], bsdf.inputs['Emission Color'])
    bsdf.inputs['Emission Strength'].default_value = 0.5 
    
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # 객체에 재질 할당
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    print(f"🎨 히트맵 재질 적용 완료 (Target Attribute: {target_attr_name})")

def update_colors_per_frame(scene):
    """
    [핸들러] 프레임 변경 시 실행됨.
    캐싱된 색상 데이터를 메쉬 속성에 덮어씌움.
    """
    obj = bpy.data.objects.get("ClothMesh")
    if not obj or not COLOR_CACHE:
        return
    
    # 현재 프레임 (1-based index라고 가정하고 0-based로 변환)
    frame_idx = scene.frame_current - 1
    
    if frame_idx in COLOR_CACHE:
        # Mesh의 활성 컬러 속성 찾기
        color_layer = None
        for name in ["Color", "Col"]:
            if name in obj.data.attributes:
                color_layer = obj.data.attributes[name]
                break
        
        # 만약 이름으로 못 찾으면 첫 번째 FLOAT_COLOR 속성 사용
        if not color_layer:
             for attr in obj.data.attributes:
                 if attr.data_type in {'FLOAT_COLOR', 'BYTE_COLOR'}:
                     color_layer = attr
                     break
        
        if color_layer:
            # [고속 업데이트] C레벨 함수 foreach_set 사용
            try:
                color_layer.data.foreach_set("color", COLOR_CACHE[frame_idx])
            except Exception as e:
                # 가끔 버텍스 수가 안 맞거나 하면 에러 날 수 있음
                pass

def import_simulation_complete(obj_dir):
    global COLOR_CACHE
    
    # 1. 파일 목록 로드
    if not os.path.exists(obj_dir):
        print(f"❌ 경로를 찾을 수 없습니다: {obj_dir}")
        return

    obj_files = [f for f in os.listdir(obj_dir) if f.lower().endswith('.obj')]
    obj_files = sort_obj_files_naturally(obj_files)
    
    if not obj_files:
        print("❌ OBJ 파일이 없습니다.")
        return
        
    total_frames = len(obj_files)
    print(f"🚀 총 {total_frames} 프레임 임포트 시작...")
    
    # 씬 설정
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = total_frames
    scene.frame_set(1)

    # -----------------------------------------------
    # 2. Base Mesh (첫 프레임) 임포트
    # -----------------------------------------------
    bpy.ops.wm.obj_import(filepath=os.path.join(obj_dir, obj_files[0]))
    base_obj = bpy.context.selected_objects[0]
    base_obj.name = 'ClothMesh'
    bpy.context.view_layer.objects.active = base_obj
    
    # Smooth Shade 적용
    bpy.ops.object.shade_smooth()
    
    # 카메라 포커싱
    focus_camera_on_object(base_obj)

    # 첫 프레임 색상 캐싱
    # Blender Importer는 OBJ의 v r g b를 'Color'라는 속성으로 가져옵니다.
    color_attr = base_obj.data.attributes.get("Color") or base_obj.data.attributes.get("Col")
    
    if color_attr:
        # 데이터 읽기 (Flattened array)
        data_len = len(color_attr.data) * 4 # RGBA per element
        colors = [0.0] * data_len
        color_attr.data.foreach_get("color", colors)
        COLOR_CACHE[0] = colors
    else:
        print("⚠️ 경고: 첫 OBJ에 색상 정보가 없거나 속성이 생성되지 않았습니다.")

    # Shape Key Basis 생성
    if not base_obj.data.shape_keys:
        base_obj.shape_key_add(name='Basis')

    # -----------------------------------------------
    # 3. 나머지 프레임 임포트 (Shape Key + Color Cache)
    # -----------------------------------------------
    # 성능을 위해 View Layer 업데이트 일시 중지 가능하지만, 안전하게 진행
    
    for i, f in enumerate(obj_files[1:], start=1):
        if i % 10 == 0:
            print(f"Processing frame {i}/{total_frames}...")
            
        filepath = os.path.join(obj_dir, f)
        bpy.ops.wm.obj_import(filepath=filepath)
        temp_obj = bpy.context.selected_objects[0]
        
        # [A] Shape Key 생성 (위치 정보 복사)
        sk = base_obj.shape_key_add(name=f'Frame_{i:04d}')
        
        verts_src = temp_obj.data.vertices
        verts_dst = sk.data
        
        # 버텍스 수 일치 확인
        if len(verts_src) == len(verts_dst):
            count = len(verts_src) * 3
            coords = [0.0] * count
            verts_src.foreach_get('co', coords)
            verts_dst.foreach_set('co', coords)
        else:
            print(f"⚠️ Vertex Count Mismatch at frame {i}")

        # [B] 색상 데이터 캐싱 (히트맵용)
        temp_attr = temp_obj.data.attributes.get("Color") or temp_obj.data.attributes.get("Col")
        if temp_attr:
            count = len(temp_attr.data) * 4
            colors = [0.0] * count
            temp_attr.data.foreach_get("color", colors)
            COLOR_CACHE[i] = colors
            
        # 임시 객체 삭제
        bpy.data.objects.remove(temp_obj, do_unlink=True)

    # 4. 애니메이션 키프레임 설정
    animate_shape_keys(base_obj, total_frames)
    
    # 5. 재질 및 조명 설정
    setup_heatmap_material(base_obj)
    setup_lighting()
    
    # 6. 핸들러 등록 (실시간 색상 업데이트)
    bpy.app.handlers.frame_change_post.append(update_colors_per_frame)
    
    print("✅ 모든 작업 완료! Spacebar를 눌러 재생하세요.")

# ==========================================
# [Main Execution]
# ==========================================
if __name__ == "__main__":
    # ▼▼▼ 여기에 실제 OBJ 폴더 경로를 넣으세요 ▼▼▼
    target_dir = r"C:\Users\NCC\Desktop\NJ\개인\cloth-python\output_frames_self_collision_v3"
    
    clear_scene()
    import_simulation_complete(target_dir)