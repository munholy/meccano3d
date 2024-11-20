import blenderproc as bproc
import argparse
import os
import numpy as np
import glob
import bpy

OBJ_SCALING = 2
NUM_VIEW = 20

parser = argparse.ArgumentParser()
parser.add_argument('stl_root_path', default="parts", help="Path to the root directory containing STL files")
parser.add_argument('cc_textures_path', nargs='?', default="texture", help="Path to downloaded cc textures")
parser.add_argument('output_dir', nargs='?', default="output", help="Path to where the final files will be saved ")
args = parser.parse_args()

def find_stl_files(root_dir):
    """Recursively find all .stl files in the given directory"""
    return glob.glob(os.path.join(root_dir, "**/*.stl"), recursive=True)


def clean_scene():
    """Clean up the scene by removing all objects except camera and lights"""
    for obj in bpy.data.objects:
        if obj.type not in ['CAMERA', 'LIGHT']:
            bpy.data.objects.remove(obj, do_unlink=True)
    # Also clean up meshes, materials, etc.
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh, do_unlink=True)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat, do_unlink=True)

bproc.init()

# First clean the scene
clean_scene()

field_of_view = np.random.uniform(0.7, 2.0)
length_scale = max(1, 2 * np.tan(field_of_view / 2))
print(field_of_view, 'fov')
print(f'{length_scale=}')

# Find and load all STL files
stl_files = find_stl_files(args.stl_root_path)
loaded_objects = []

num_obj_sample = np.random.randint(10,30)
stl_files = np.random.choice(stl_files, num_obj_sample, replace = True).tolist()

for i, stl_file in enumerate(stl_files):
    # Load STL file
    obj = bproc.loader.load_obj(stl_file)[0]
    # Scale object to reasonable size (adjust scale factor as needed)

    rand_scale = 3.0 ** np.random.random()

    # 바운딩 박스 기반으로 크기 계산
    bbox = obj.get_bound_box()
    dims = np.ptp(bbox, axis=0)  # 각 축의 최대-최소 차이
    longest_axis = max(np.max(dims), 0.00001)
    size_thres = 20.0 * 5.0 ** np.random.random() # 20mm ~ 100mm
    target_size = min(size_thres, longest_axis)
        

    size_scale = target_size / longest_axis

    # 최종 스케일 적용
    final_scale = 0.001 * rand_scale * max(1,length_scale) * OBJ_SCALING * size_scale
    obj.set_scale([final_scale] * 3)
    # obj.set_scale([0.001 * rand_scale * max(1,length_scale) * OBJ_SCALING] * 3)  # Convert mm to m
    
    # Set category_id for BOP format
    obj.set_cp("category_id", i + 1)  # BOP format requires category_id
    obj.set_cp("bop_dataset_name", "meccano3d")  # Set custom dataset name
    
    loaded_objects.append(obj)

# Set shading and physics properties and randomize PBR materials
for obj in loaded_objects:
    obj.enable_rigidbody(True, friction=100.0, linear_damping=0.99, angular_damping=0.99)
    obj.set_shading_mode('auto')
    
    # Create new material for each object
    mat = bproc.material.create('custom_mat')
    # Randomize material properties
    base_color = np.random.uniform([0.1, 0.1, 0.1, 1.0], [1.0, 1.0, 1.0, 1.0]) 
    if np.random.random() > 0.5:
        base_color[:3] = np.random.uniform(0.1,1.0)
    mat.set_principled_shader_value("Base Color", base_color)
    mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
    mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0, 1.0))
    mat.set_principled_shader_value("Metallic", np.random.uniform(0, 1.0))
    obj.replace_materials(mat)

# Create room
wall_pos = np.random.random(size = (8,)) + 2.0
wall_angle = np.random.uniform(low = -np.pi / 8, high = np.pi / 8, size = (8,))
room_planes = [bproc.object.create_primitive('PLANE', scale=[5, 5, 1]),
               bproc.object.create_primitive('PLANE', scale=[5, 5, 1], location=[0, -wall_pos[0], wall_pos[1]], rotation=[-1.570796, wall_angle[0],wall_angle[1]]),
               bproc.object.create_primitive('PLANE', scale=[5, 5, 1], location=[0, wall_pos[2], wall_pos[3]], rotation=[1.570796, wall_angle[2],wall_angle[3]]),
               bproc.object.create_primitive('PLANE', scale=[5, 5, 1], location=[wall_pos[4], 0, wall_pos[5]], rotation=[wall_angle[4], -1.570796, wall_angle[5]]),
               bproc.object.create_primitive('PLANE', scale=[5, 5, 1], location=[-wall_pos[6], 0, wall_pos[7]], rotation=[wall_angle[6], 1.570796, wall_angle[7]])]

for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', friction=100.0, linear_damping=0.99, angular_damping=0.99)

# Sample light color and strength from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                 emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))    
light_plane.replace_materials(light_plane_material)

# Sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(np.random.uniform(100,200))
light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
location = bproc.sampler.shell(center=[0, 0, 0], radius_min=1, radius_max=1.5,
                             elevation_min=5, elevation_max=89, uniform_volume=False)
light_point.set_location(location)

# Sample CC Texture and assign to room planes
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)
random_cc_texture = np.random.choice(cc_textures)
for plane in room_planes:
    plane.replace_materials(random_cc_texture)



# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0]) * length_scale
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6]) * length_scale
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

# Sample object poses and check collisions 
bproc.object.sample_poses(objects_to_sample=loaded_objects,
                         sample_pose_func=sample_pose_func, 
                         max_tries=1000)

# Physics Positioning
bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                max_simulation_time=10,
                                                check_object_interval=1,
                                                substeps_per_frame=20,
                                                solver_iters=25)

# Filter objects by their positions
filtered_objects = []
for obj in loaded_objects:
    pos = np.mean(obj.get_bound_box(), axis=0)
    if (-1.0 <= pos[0] <= 1.0 and    # x 범위
        -1.0 <= pos[1] <= 1.0 and    # y 범위
        -0.05 <= pos[2] <= 0.5):     # z 범위
        filtered_objects.append(obj)
    else:
        obj.hide()
        obj.delete()  # 범위를 벗어난 객체 삭제

loaded_objects = filtered_objects  # 필터링된 객체 목록으로 업데이트
print(len(loaded_objects), 'filtered num')

# BVH tree used for camera obstacle checks
bvh_tree = bproc.object.create_bvh_tree_multi_objects(loaded_objects)

image_width, image_height = 512,512
if np.random.random() > 0.5:
    image_width, image_height = 720,480
if np.random.random() > 0.5:
    image_width, image_height = 640,480

bproc.camera.set_resolution(image_width=image_width, image_height=image_height)

bproc.camera.set_intrinsics_from_blender_params(lens=field_of_view, lens_unit="FOV")

# Sample camera poses
poses = 0
while poses < NUM_VIEW:
    # Sample location
    location = bproc.sampler.shell(center=[0, 0, 0],
                                 radius_min=0.61,
                                 radius_max=1.24,
                                 elevation_min=5,
                                 elevation_max=89,
                                 uniform_volume=False)
    # Determine point of interest in scene as the object closest to the mean of a subset of objects
    # poi = bproc.object.compute_poi(np.random.choice(loaded_objects, size=min(10, len(loaded_objects))))

    # Custom PoI logic
    num_obj_poi = np.random.randint(low = 4, high = num_obj_sample // 2)
    objects_poi = np.random.choice(loaded_objects, size=max(1, num_obj_poi))
    mean_bb_points = [np.mean(obj.get_bound_box(), axis = 0) for obj in objects_poi]
    # print(mean_bb_points)
    # Query point - mean of means
    mean_bb_point = np.mean(mean_bb_points, axis=0)
    # print(mean_bb_point)
    poi = mean_bb_point


    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    
    # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
    if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bvh_tree):
        # Persist camera pose
        bproc.camera.add_camera_pose(cam2world_matrix)
        poses += 1

# Activate depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

# Render the whole pipeline
data = bproc.renderer.render()

# Write data in bop format
bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                      target_objects=loaded_objects,
                      dataset="meccano3d",
                      depths=data["depth"],
                      colors=data["colors"], 
                      color_file_format="JPEG",
                      depth_scale=0.1,  # Adjust this value based on your depth range
                      jpg_quality=90,
                      save_world2cam=True,  # Save camera poses
                      append_to_existing_output=True,
                      frames_per_chunk=1000,
                      m2mm=True,  # Convert units from m to mm
                      calc_mask_info_coco=True)  # Calculate object masks and COCO format annotations
