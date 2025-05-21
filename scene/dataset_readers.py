#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import math
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from scene.torf_utils import *
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB, RGB2SH, SH2PA, PA2SH
from scene.gaussian_model import BasicPointCloud
from tqdm import tqdm

class CameraInfo(NamedTuple): 
    uid: int
    # Color
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_name: str
    image: np.array
    image_path: str
    width: int
    height: int
    frame_id: Optional[int] = -1
    # Time-of-Flight
    R_tof: Optional[np.array] = None
    T_tof: Optional[np.array] = None
    FovY_tof: Optional[np.array] = None
    FovX_tof: Optional[np.array] = None
    tof_image_name: Optional[str] = ""
    tof_image: Optional[np.array] = None
    tof_image_path: Optional[str] = ""
    distance_image_name: Optional[str] = ""     # Distance image (for synthetic scenes)
    distance_image: Optional[np.array] = None
    distance_image_path: Optional[str] = ""
    tof_width: Optional[int] = -1
    tof_height: Optional[int] = -1
    # Raw ToF Quads
    tofQuad0_im: Optional[np.array] = None
    tofQuad1_im: Optional[np.array] = None
    tofQuad2_im: Optional[np.array] = None
    tofQuad3_im: Optional[np.array] = None
    dc_offset: Optional[np.array] = np.array(0.0).astype(np.float32)
    forward_flow: Optional[np.array] = None
    backward_flow: Optional[np.array] = None
    # Others
    znear: Optional[float] = 0.01
    zfar: Optional[float] = 100.0
    depth_range: Optional[float] = 15.0
    phase_offset: Optional[np.array] = np.array(0.0).astype(np.float32)
    color_motion_mask: Optional[np.array] = None
    tof_motion_mask: Optional[np.array] = None
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    fx_tof: Optional[float] = None
    fy_tof: Optional[float] = None
    cx_tof: Optional[float] = None
    cy_tof: Optional[float] = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    spiral_cameras: Optional[list] = []

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T) # No change at all
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate.tolist(), "radius": radius.item()}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    phases, amplitudes = None, None
    if 'phase' in vertices.data.dtype.names and 'amplitude' in vertices.data.dtype.names:
        phases = np.vstack([vertices['phase']]).T
        amplitudes = np.vstack([vertices['amplitude']]).T
    seg_colors = None
    if 'seg_red' in vertices.data.dtype.names and 'seg_green' in vertices.data.dtype.names and 'seg_blue' in vertices.data.dtype.names:
        seg_colors = np.vstack([vertices['seg_red'], vertices['seg_green'], vertices['seg_blue']]).T / 255.0

    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals, phases=phases, amplitudes=amplitudes, seg_colors=seg_colors)

def storePly(path, xyz, colors, phases=None, amplitudes=None, seg_colors=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    
    normals = np.zeros_like(xyz)
    attributes = np.concatenate((xyz, normals), axis=1)
    
    dtype += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    attributes = np.concatenate((attributes, colors), axis=1)
    if phases is not None and amplitudes is not None:
        dtype += [('phase', 'f4'), ('amplitude', 'f4')]
        attributes = np.concatenate((attributes, phases, amplitudes), axis=1)
    if seg_colors is not None:
        dtype += [('seg_red', 'u1'), ('seg_green', 'u1'), ('seg_blue', 'u1')]
        attributes = np.concatenate((attributes, seg_colors), axis=1)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# COLMAP scenes
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readColmapSceneInfo(path, images, eval, llffhold=8):
    # For detailed explanation, refer to https://colmap.github.io/format.html#images-txt
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# NeRF synthetic scenes
def readCamerasFromTransforms(path, transformsfile, bg_color, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg_color * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            distance_image_name = frame["file_path"].split('/')[-1]
            distance_image_path, distance_image = "", None
            for root, _, files in os.walk(os.path.join(path, frame["file_path"].split('/')[-2])):
                for file in files:
                    if distance_image_name in file and "depth" in file:
                        distance_image_path = os.path.join(root, file)
                        break
            if distance_image_path != "":
                distance_image = Image.open(distance_image_path)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, 
                                        image=image, image_path=image_path, image_name=image_name, 
                                        distance_image_name=distance_image_name, distance_image_path=distance_image_path, distance_image=distance_image,
                                        width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, bg_color, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", bg_color[0], extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", bg_color[0], extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if os.path.exists(ply_path):
        os.remove(ply_path)
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

        # Init color and phasor
        shs_color = RGB2SH(np.ones((num_pts, 3)) * 0.5)
        colors = SH2RGB(shs_color)
        shs_phase = PA2SH(np.random.random((num_pts, 1)) * 2.0 * np.pi)
        shs_amp = PA2SH(np.ones((num_pts, 1)) * 0.5)
        phases = SH2PA(shs_phase) 
        amplitudes = SH2PA(shs_amp)

        seg_colors = np.repeat(np.array([[0.0, 0.0, 0.0]]), num_pts, axis=0) # All static
        pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros((num_pts, 3)), phases=phases, amplitudes=amplitudes, seg_colors=seg_colors)

        colors *= 255.0
        seg_colors *= 255.0
        storePly(ply_path, xyz, colors, phases, amplitudes, seg_colors)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# ToRF scenes
def readToRFCameras(path, tof_extrinsics, tof_intrinsics, color_extrinsics, color_intrinsics, depth_range, phase_offset, znear, zfar, args):
    color_images, tof_images = [], []
    for fid in tqdm(range(args.total_num_views), desc="Loading all images"):
        color_image_name = f"{fid:04d}"
        color_image_path = os.path.join(path, "color", f"{color_image_name}.npy")
        color_images.append(scale_image(np.load(color_image_path), args.color_scale_factor))
        tof_image_name = f"{fid:04d}"
        tof_image_path = os.path.join(path, "tof", f"{tof_image_name}.npy")
        tof_images.append(scale_image(np.load(tof_image_path), args.tof_scale_factor))
    color_images = normalize_im_max(np.stack(color_images)).astype(np.float32)
    tof_images = normalize_im_max(np.stack(tof_images)).astype(np.float32)

    cam_infos = []
    for fid in tqdm(range(args.total_num_views), desc="Loading all views/frames"):
        # Color camera
        R = np.transpose(color_extrinsics[fid, :3, :3]) # torf extrinsics is w2c
        T = color_extrinsics[fid, :3, 3]
        fx, fy = color_intrinsics[fid][0, 0], color_intrinsics[fid][1, 1]
        cx, cy = color_intrinsics[fid][0, 2], color_intrinsics[fid][1, 2]
        FovY = 2 * np.arctan2(args.color_image_height, 2 * fy) # radian
        FovX = 2 * np.arctan2(args.color_image_width, 2 * fx)
        
        color_image_name = f"{fid:04d}"
        color_image_path = os.path.join(path, "color", f"{color_image_name}.npy")
        color_image = Image.fromarray(np.array(color_images[fid] * 255.0, dtype=np.byte), "RGB")

        # Time-of-Flight camera
        R_tof = np.transpose(tof_extrinsics[fid, :3, :3])
        T_tof = tof_extrinsics[fid, :3, 3]
        fx_tof, fy_tof = tof_intrinsics[fid][0, 0], tof_intrinsics[fid][1, 1]
        cx_tof, cy_tof = tof_intrinsics[fid][0, 2], tof_intrinsics[fid][1, 2]
        FovY_tof = 2 * np.arctan2(args.tof_image_height, 2 * fy_tof)
        FovX_tof = 2 * np.arctan2(args.tof_image_width, 2 * fx_tof)

        tof_image_name = f"{fid:04d}"
        tof_image_path = os.path.join(path, "tof", f"{tof_image_name}.npy")
        tof_image = tof_images[fid]
        
        # Distance image (for synthetic scenes)        
        distance_image_name = f"{fid:04d}" 
        distance_image_path = os.path.join(path, "distance", f"{distance_image_name}.npy")
        distance_image = scale_image(np.load(distance_image_path), args.tof_scale_factor, interpolation=cv2.INTER_NEAREST)

        color_motion_mask, tof_motion_mask = None, None
        if args.dynamic and os.path.exists(os.path.join(path, "mask_color")):
            color_motion_mask = np.load(os.path.join(path, "mask_color", f"{color_image_name}.npy")).astype(np.float32) / 255.0
            tof_motion_mask = np.load(os.path.join(path, "mask_tof", f"{tof_image_name}.npy")).astype(np.float32) / 255.0

        frame_id = fid if "dino" not in path else fid % 61
        cam_infos.append(CameraInfo(
            uid=fid, frame_id=frame_id, 
            # Color
            R=R, T=T, FovY=FovY, FovX=FovX, 
            fx=fx*args.color_scale_factor, fy=fy*args.color_scale_factor, cx=cx*args.color_scale_factor, cy=cy*args.color_scale_factor,
            image_name=color_image_name, image=color_image, image_path=color_image_path,
            width=int(args.color_image_width*args.color_scale_factor), height=int(args.color_image_height*args.color_scale_factor),
            # Time-of-Flight
            R_tof=R_tof, T_tof=T_tof, FovY_tof=FovY_tof, FovX_tof=FovX_tof, 
            fx_tof=fx_tof*args.tof_scale_factor, fy_tof=fy_tof*args.tof_scale_factor, cx_tof=cx_tof*args.tof_scale_factor, cy_tof=cy_tof*args.tof_scale_factor,
            tof_image_name=tof_image_name, tof_image=tof_image, tof_image_path=tof_image_path,
            distance_image_name=distance_image_name, distance_image=distance_image, distance_image_path=distance_image_path,
            tof_width=int(args.tof_image_width*args.tof_scale_factor), tof_height=int(args.tof_image_height*args.tof_scale_factor),
            # Others
            znear=znear, zfar=zfar, depth_range=depth_range, phase_offset=phase_offset,
            color_motion_mask=color_motion_mask, tof_motion_mask=tof_motion_mask))
    return cam_infos

def readToRFSpiralCameras(extrinsics, intrinsics, depth_range, phase_offset, znear, zfar, args):
    cam_infos = []
    
    for fid in tqdm(range(args.total_num_spiral_views)):
        R = np.transpose(extrinsics[fid, :3, :3])
        T = extrinsics[fid, :3, 3]
        fx, fy = intrinsics[fid][0, 0], intrinsics[fid][1, 1]
        cx, cy = intrinsics[fid][0, 2], intrinsics[fid][1, 2]
        FovY = 2 * np.arctan2(args.color_image_height, 2 * fy)
        FovX = 2 * np.arctan2(args.color_image_width, 2 * fx)
        
        cam_infos.append(CameraInfo(uid=fid, frame_id=fid,
                                    R=R, T=T, FovY=FovY, FovX=FovX, fx=fx*args.color_scale_factor, fy=fy*args.color_scale_factor, cx=cx*args.color_scale_factor, cy=cy*args.color_scale_factor,
                                    image_name=f"{fid:04d}", image=None, image_path=None,
                                    width=int(args.color_image_width*args.color_scale_factor), height=int(args.color_image_height*args.color_scale_factor), 
                                    R_tof=R, T_tof=T, FovY_tof=FovY, FovX_tof=FovX, fx_tof=fx*args.tof_scale_factor, fy_tof=fy*args.tof_scale_factor, cx_tof=cx*args.tof_scale_factor, cy_tof=cy*args.tof_scale_factor,
                                    tof_image_name=f"{fid:04d}", tof_image=None, tof_image_path=None,
                                    tof_width=int(args.tof_image_width*args.tof_scale_factor), tof_height=int(args.tof_image_height*args.tof_scale_factor), 
                                    znear=znear, zfar=zfar, depth_range=depth_range, phase_offset=phase_offset))
    return cam_infos

def readToRFSceneInfo(path, eval, args, llffhold=8):
    # Load cameras
    if args.dataset_type == "real":
        cam_file_ending = 'mat'
    else:
        cam_file_ending = 'npy'

    tof_intrinsics, tof_extrinsics = get_camera_params(
        os.path.join(path, 'cams', f'tof_intrinsics.{cam_file_ending}'), 
        os.path.join(path, 'cams', 'tof_extrinsics.npy'), 
        args.total_num_views)
    color_intrinsics, color_extrinsics = get_camera_params(
        os.path.join(path, 'cams', f'color_intrinsics.{cam_file_ending}'), 
        os.path.join(path, 'cams', 'color_extrinsics.npy'), 
        args.total_num_views)
    
    relative_pose = os.path.join(path, 'cams', 'relative_pose.npy')
    if os.path.exists(relative_pose):
        E = np.load(relative_pose)
        color_extrinsics = np.linalg.inv(E) @ tof_extrinsics

    phase_offset_path = os.path.join(path, 'cams', 'phase_offset.npy')
    if args.phase_offset != -99.0:
        phase_offset = np.array(args.phase_offset).astype(np.float32)
    elif os.path.exists(phase_offset_path):
        phase_offset = np.load(phase_offset_path).astype(np.float32)
    else:
        phase_offset = np.array(0.0).astype(np.float32)
    
    depth_range_path = os.path.join(path, 'cams', 'depth_range.npy')
    if os.path.exists(depth_range_path):
        depth_range = np.load(depth_range_path).astype(np.float32)
    else:
        depth_range = np.array(args.depth_range).astype(np.float32)
    znear = args.min_depth_fac * depth_range * 0.9
    zfar = args.max_depth_fac * depth_range * 1.1
    
    # Create splits
    cam_infos_unsorted = readToRFCameras(path, tof_extrinsics, tof_intrinsics, color_extrinsics, color_intrinsics, depth_range, phase_offset, znear, zfar, args)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if not args.dynamic and eval:
        if args.train_views != "":
            idx_train = [int(i.strip()) for i in args.train_views.split(",")]
            idx_test = [i for i in np.arange(args.total_num_views) if (i not in idx_train)] 
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in idx_train]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in idx_test]
        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    elif "dino" in path and eval:
        train_cam_infos = cam_infos[:30]
        test_cam_infos = cam_infos[len(cam_infos)//2:len(cam_infos)//2+30]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if nerf_normalization["radius"] == 0.0:
        nerf_normalization["radius"] = 1.0
    nerf_normalization['scene_scale'] = depth_range.item() * 0.55

    test_poses_path = os.path.join(path, 'test_poses.npy')
    if os.path.exists(test_poses_path):
        temp_pose = np.load(test_poses_path)
        split_pose = np.tile(np.eye(4)[None], (temp_pose.shape[0], 1, 1))
        split_pose[:, :3, :] = temp_pose[:, :3, :4]
        split_pose = np.linalg.inv(split_pose)
        split_pose[:, :3, -1] *= 1.1
        split_pose, _ = recenter_poses(split_pose)
        split_pose = split_pose[::-1]
        spiral_poses = split_pose
    else:
        all_poses = [np.linalg.inv(ext) for ext in np.concatenate([tof_extrinsics], axis=0)]
        if not args.dynamic:
            spiral_poses = get_render_poses_spiral(-1.0, np.array([znear, zfar]), all_poses, N_views=args.total_num_spiral_views, N_rots=1)[::-1]
        else:
            spiral_poses = get_render_poses_spiral(-1.0, np.array([znear, zfar]), all_poses, N_views=args.total_num_spiral_views)
    spiral_exts = np.array([np.linalg.inv(pose) for pose in spiral_poses])
    spiral_intrinsics = [np.copy(color_intrinsics[0]) for _ in range(spiral_exts.shape[0])]
    spiral_cam_infos = readToRFSpiralCameras(spiral_exts, spiral_intrinsics, depth_range, phase_offset, znear, zfar, args)

    # Initialize point cloud
    min_bounds, max_bounds = calculateSceneBounds(train_cam_infos, args)
    ply_path = os.path.join(args.model_path, "points3d.ply")
    if args.init_method == "random":
        total_num_pts = args.num_points
        print(f"Generating random point cloud ({total_num_pts})...")

        # Init xyz
        xyz_all = np.random.uniform(min_bounds, max_bounds, (total_num_pts, 3))
       
        # Init color and phasor
        shs_color = RGB2SH(np.ones((total_num_pts, 3)) * 0.5)
        colors = SH2RGB(shs_color)
        shs_phase = PA2SH(np.random.random((total_num_pts, 1)) * 2.0 * np.pi)
        shs_amp = PA2SH(np.ones((total_num_pts, 1)) * args.initial_amplitude)
        phases = SH2PA(shs_phase) 
        amplitudes = SH2PA(shs_amp)
    elif args.init_method == "phase":
        if args.dynamic:
            cam_fid_list = [args.total_num_views // 2]
        else:
            cam_fid_list = [idx for idx in range(len(train_cam_infos))]
        
        xyz_all, shs_amp_all, shs_color_all = [], [], []
        for fid in cam_fid_list:
            tof_cam = train_cam_infos[fid]
            tof_depth_height = math.ceil(tof_cam.tof_height / args.phase_resolution_stride)
            tof_depth_width = math.ceil(tof_cam.tof_width / args.phase_resolution_stride)

            # Pixel space
            xy_screen = np.indices((tof_depth_height, tof_depth_width)).transpose(1, 2, 0).reshape(-1, 2).astype(np.float32)[:, ::-1] * args.phase_resolution_stride
            xy_screen = xy_screen.astype(np.int16)
            xy_screen = np.concatenate([xy_screen, xy_screen], axis=0)

            num_pts = xy_screen.shape[0]
            xyzw = np.empty((num_pts, 4))
            view_mat = getWorld2View2(tof_cam.R_tof, tof_cam.T_tof)

            # Normalize to [-WInMeters/2, WInMeters/2] and [-HInMeters/2, HInMeters/2]
            WInMeters = tof_cam.znear * np.tan(tof_cam.FovX_tof / 2.0) * 2.0
            HInMeters = tof_cam.znear * np.tan(tof_cam.FovY_tof / 2.0) * 2.0
            xyzw[:, 0] = (xy_screen[:, 0] * 2.0 / tof_cam.tof_width - 1.0) * WInMeters / 2.0
            xyzw[:, 1] = (xy_screen[:, 1] * 2.0 / tof_cam.tof_height - 1.0) * HInMeters / 2.0

            # Distances to Light
            z = depth_from_tof(tof_cam.tof_image[xy_screen[:, 1], xy_screen[:, 0], :], depth_range, phase_offset).reshape(num_pts, 1) 
            z[num_pts:, 0] += depth_range / 2.0

            # Camera space.
            dists2pixInMeters = np.sqrt(np.square(xyzw[:, 0]) + np.square(xyzw[:, 1]) + np.square(tof_cam.znear))
            np.true_divide(xyzw[:, 0], dists2pixInMeters, out=xyzw[:, 0]) 
            np.true_divide(xyzw[:, 1], dists2pixInMeters, out=xyzw[:, 1])
            np.multiply(xyzw[:, 0:1], z, out=xyzw[:, 0:1])
            np.multiply(xyzw[:, 1:2], z, out=xyzw[:, 1:2])
            xyzw[:, 2:3] = np.sqrt(np.square(z) - np.square(xyzw[:, 0:1]) - np.square(xyzw[:, 1:2]))
            xyzw[:, 3:4] = np.ones((num_pts, 1))

            # World space.
            xyz = (np.linalg.inv(view_mat) @ xyzw.T).T[:, :3]
            shs_color = RGB2SH(tof_cam.tof_image[xy_screen[:, 1], xy_screen[:, 0], 2].reshape(-1, 1) * np.ones((1, 3), dtype=np.float32))
            shs_amp = PA2SH(tof_cam.tof_image[xy_screen[:, 1], xy_screen[:, 0], 2].reshape(-1, 1) * np.square(z))
            xyz_all.append(xyz), shs_amp_all.append(shs_amp), shs_color_all.append(shs_color)

        xyz_all = np.concatenate(xyz_all, axis=0)
        shs_amp_all = np.concatenate(shs_amp_all, axis=0)
        shs_color_all = np.concatenate(shs_color_all, axis=0)

        total_num_pts = xyz_all.shape[0]
        print(f"Generating point cloud based on depth from the canonical frame ({total_num_pts})...")

        colors = SH2RGB(shs_color_all)
        shs_phase = PA2SH(np.zeros((total_num_pts, 1)).astype(np.float32))
        phases = SH2PA(shs_phase) 
        amplitudes = SH2PA(shs_amp_all)

    seg_colors = np.repeat(np.array([[1.0, 0.0, 0.0]]), total_num_pts, axis=0) # All dynamic
    pcd = BasicPointCloud(points=xyz_all, colors=colors, normals=np.zeros((total_num_pts, 3)), phases=phases, amplitudes=amplitudes, seg_colors=seg_colors)

    colors *= 255.0
    seg_colors *= 255.0
    storePly(ply_path, xyz_all, colors, phases, amplitudes, seg_colors)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           spiral_cameras=spiral_cam_infos)
    return scene_info

def readToRFDepthMaps(path, model_path, iteration, args):
    # Load cameras
    if args.dataset_type == "real":
        cam_file_ending = 'mat'
    else:
        cam_file_ending = 'npy'

    tof_intrinsics, tof_extrinsics = get_camera_params(
        os.path.join(path, 'cams', f'tof_intrinsics.{cam_file_ending}'), 
        os.path.join(path, 'cams', 'tof_extrinsics.npy'), 
        args.total_num_views)
    color_intrinsics, color_extrinsics = get_camera_params(
        os.path.join(path, 'cams', f'color_intrinsics.{cam_file_ending}'), 
        os.path.join(path, 'cams', 'color_extrinsics.npy'), 
        args.total_num_views)
    
    relative_pose = os.path.join(path, 'cams', 'relative_pose.npy')
    if os.path.exists(relative_pose):
        E = np.load(relative_pose)
        color_extrinsics = np.linalg.inv(E) @ tof_extrinsics

    phase_offset_path = os.path.join(path, 'cams', 'phase_offset.npy')
    if os.path.exists(phase_offset_path):
        phase_offset = np.load(phase_offset_path).astype(np.float32)
    else:
        phase_offset = np.array(args.phase_offset).astype(np.float32)
    
    depth_range_path = os.path.join(path, 'cams', 'depth_range.npy')
    if os.path.exists(depth_range_path):
        depth_range = np.load(depth_range_path).astype(np.float32)
    else:
        depth_range = np.array(args.depth_range).astype(np.float32)
    znear = args.min_depth_fac * depth_range * 0.9
    zfar = args.max_depth_fac * depth_range * 1.1
    
    # Create splits
    cam_infos_unsorted = readFToRFCameras(path, tof_extrinsics, tof_intrinsics, color_extrinsics, color_intrinsics, depth_range, phase_offset, znear, zfar, args)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    nerf_normalization = getNerfppNorm(cam_infos)
    if nerf_normalization["radius"] == 0.0:
        nerf_normalization["radius"] = 1.0
    nerf_normalization['scene_scale'] = depth_range.item() * 0.55

    rendered_depth_images = {}
    for fp in os.listdir(os.path.join(model_path, f"ours_{iteration}", "renders", "depth_norm_tof_cam")):
        if fp.endswith(".npy"):
            idx = int(fp.split(".")[0])
            depth_tof_cam_im = np.load(os.path.join(model_path, f"ours_{iteration}", "renders", "depth_norm_tof_cam", fp))
            rendered_depth_images[idx] = depth_tof_cam_im

    # Initialize point clouds
    for fid in range(len(cam_infos)):
        os.makedirs(os.path.join(model_path, "proxy_pcd", f"frame_{fid}", "point_cloud", f"iteration_{iteration}"), exist_ok=True)
        ply_path = os.path.join(model_path, "proxy_pcd", f"frame_{fid}", f"input.ply")

        tof_cam = cam_infos[fid]
        tof_depth_height = math.ceil(tof_cam.tof_height)
        tof_depth_width = math.ceil(tof_cam.tof_width)

        # Pixel space
        xy_screen = np.indices((tof_depth_height, tof_depth_width)).transpose(1, 2, 0).reshape(-1, 2).astype(np.float32)[:, ::-1]
        xy_screen = xy_screen.astype(np.int16)

        num_pts = xy_screen.shape[0]
        xyzw = np.empty((num_pts, 4))
        view_mat = getWorld2View2(tof_cam.R_tof, tof_cam.T_tof)

        # Normalize to [-WInMeters/2, WInMeters/2] and [-HInMeters/2, HInMeters/2]
        WInMeters = tof_cam.znear * np.tan(tof_cam.FovX_tof / 2.0) * 2.0
        HInMeters = tof_cam.znear * np.tan(tof_cam.FovY_tof / 2.0) * 2.0
        xyzw[:, 0] = (xy_screen[:, 0] * 2.0 / tof_cam.tof_width - 1.0) * WInMeters / 2.0
        xyzw[:, 1] = (xy_screen[:, 1] * 2.0 / tof_cam.tof_height - 1.0) * HInMeters / 2.0
        xyzw = np.concatenate([xyzw, xyzw], axis=0)

        # Distances to Light
        ## input depth
        z_i = depth_from_tof(tof_cam.tof_image[xy_screen[:, 1], xy_screen[:, 0], :], depth_range, phase_offset).reshape(num_pts, 1)
        ## rendered depth
        z_r = rendered_depth_images[fid][xy_screen[:, 1], xy_screen[:, 0]].reshape(num_pts, 1) 
        ## Concatenate
        z = np.concatenate([z_i, z_r], axis=0)

        # Camera space.
        dists2pixInMeters = np.sqrt(np.square(xyzw[:, 0]) + np.square(xyzw[:, 1]) + np.square(tof_cam.znear))
        np.true_divide(xyzw[:, 0], dists2pixInMeters, out=xyzw[:, 0]) 
        np.true_divide(xyzw[:, 1], dists2pixInMeters, out=xyzw[:, 1])
        np.multiply(xyzw[:, 0:1], z, out=xyzw[:, 0:1])
        np.multiply(xyzw[:, 1:2], z, out=xyzw[:, 1:2])
        xyzw[:, 2:3] = np.sqrt(np.square(z) - np.square(xyzw[:, 0:1]) - np.square(xyzw[:, 1:2]))
        xyzw[:, 3:4] = 1

        # World space.
        xyz = (np.linalg.inv(view_mat) @ xyzw.T).T[:, :3]
        colors = np.tile([1, 0, 0], (xyz.shape[0], 1)).astype(np.float32)
        colors[xyz.shape[0]//2:, :] = np.tile([0, 0, 1], (xyz.shape[0]//2, 1)).astype(np.float32)

        phases = SH2PA(PA2SH(np.zeros((xyz.shape[0], 1)).astype(np.float32))) 
        amplitudes = SH2PA(PA2SH(np.zeros((xyz.shape[0], 1)).astype(np.float32))) 
        seg_colors = np.repeat(np.array([[0.0, 0.0, 0.0]]), xyz.shape[0], axis=0) # All dynamic
        
        colors *= 255.0
        seg_colors *= 255.0
        storePly(ply_path, xyz, colors, phases=phases, amplitudes=amplitudes, seg_colors=seg_colors)

    return len(cam_infos), cam_infos

# F-ToRF (raw quads) scene
def readFToRFCameras(path, tof_extrinsics, tof_intrinsics, color_extrinsics, color_intrinsics, depth_range, phase_offset, dc_offset, znear, zfar, quad_values_scale_factor, args):
    missing_frames = {
        'color' : set(),
        'tofType0' : set(),
        'tofType1' : set(),
        'tofType2' : set(),
        'tofType3' : set(),
    } 

    # Expected shapes are updated based on frame 0000
    color_shape = np.load(os.path.join(path, "color", f"0000.npy")).shape
    tofQuad_shape = np.load(os.path.join(path, "tofType0", f"0000.npy")).shape
    depth_shape = np.load(os.path.join(path, "synthetic_depth", f"0000.npy")).shape
    flow_shape = np.load(os.path.join(path, "forward_flow", f"flow_0000.npy")).shape
 
    color_images, tof_images = [], []
    for fid in tqdm(range(args.total_num_views), desc="Loading all images"):
        # Load color images
        color_image_name = f"{fid:04d}"
        color_image_path = os.path.join(path, "color", f"{color_image_name}.npy")
        if os.path.exists(color_image_path):
            color_image = np.load(color_image_path)
        else:
            missing_frames['color'].add(fid)
            color_image = np.zeros(color_shape, dtype=np.float32)
        color_images.append(scale_image(color_image, args.color_scale_factor))
        # Load tof images
        tof_image_name = f"{fid:04d}"
        tof_image_path = os.path.join(path, "synthetic_tof", f"{tof_image_name}.npy")
        if os.path.exists(tof_image_path):
            tof_image = np.load(tof_image_path)
        else:
            tof_image = np.zeros([tofQuad_shape[0], tofQuad_shape[1], 3], dtype=np.float32)
        tof_images.append(scale_image(tof_image, args.tof_scale_factor))
    color_images = normalize_im_max(np.stack(color_images)).astype(np.float32)
    tof_images = normalize_im_max(np.stack(tof_images)).astype(np.float32)

    cam_infos = []
    for fid in tqdm(range(args.total_num_views), desc="Loading all views/frames"):
        # Color camera
        R = np.transpose(color_extrinsics[fid, :3, :3]) # torf extrinsics is w2c
        T = color_extrinsics[fid, :3, 3]
        fx, fy = color_intrinsics[fid][0, 0], color_intrinsics[fid][1, 1]
        cx, cy = color_intrinsics[fid][0, 2], color_intrinsics[fid][1, 2]
        FovY = 2 * np.arctan2(args.color_image_height, 2 * fy) # radian
        FovX = 2 * np.arctan2(args.color_image_width, 2 * fx)
        
        color_image_name = f"{fid:04d}"
        color_image_path = os.path.join(path, "color", f"{color_image_name}.npy")
        color_image = Image.fromarray(np.array(color_images[fid] * 255.0, dtype=np.byte), "RGB")

        # Time-of-Flight camera
        R_tof = np.transpose(tof_extrinsics[fid, :3, :3])
        T_tof = tof_extrinsics[fid, :3, 3]
        fx_tof, fy_tof = tof_intrinsics[fid][0, 0], tof_intrinsics[fid][1, 1]
        cx_tof, cy_tof = tof_intrinsics[fid][0, 2], tof_intrinsics[fid][1, 2]
        FovY_tof = 2 * np.arctan2(args.tof_image_height, 2 * fy_tof)
        FovX_tof = 2 * np.arctan2(args.tof_image_width, 2 * fx_tof)

        tof_image_name = f"{fid:04d}"
        tof_image_path = os.path.join(path, "synthetic_tof", f"{tof_image_name}.npy")
        tof_image = tof_images[fid]

        # ToF Quads
        tofQuad_ims = []
        last_int_fid = (fid // 4) * 4
        for t in range(4):
            tofQuad_im_path = os.path.join(path, f"tofType{t}", f"{last_int_fid+t:04d}.npy")
            tofQuad_im = np.load(tofQuad_im_path) * quad_values_scale_factor
            tofQuad_ims.append(scale_image(tofQuad_im, args.tof_scale_factor))
        
        forward_flow, backward_flow = None, None
        forward_flow_path = os.path.join(path, f"forward_flow_2", f"flow_{fid:04d}.npy")
        backward_flow_path = os.path.join(path, f"backward_flow_2", f"flow_{fid:04d}.npy")
        if os.path.exists(forward_flow_path):
            forward_flow = np.load(forward_flow_path)
            forward_flow = scale_image(forward_flow.transpose(1, 2, 0), args.color_scale_factor)
        if os.path.exists(backward_flow_path):
            backward_flow = np.load(backward_flow_path)
            backward_flow = scale_image(backward_flow.transpose(1, 2, 0), args.color_scale_factor)
        
        # Distance image (depth from phasor)
        distance_image_name = f"{fid:04d}" 
        distance_image_path = os.path.join(path, "synthetic_depth", f"{distance_image_name}.npy")
        if os.path.exists(distance_image_path):
            distance_image = np.load(distance_image_path)
        else:
            distance_image = np.zeros(depth_shape, dtype=np.float32)
        distance_image = scale_image(distance_image, args.tof_scale_factor, interpolation=cv2.INTER_NEAREST)

        # tof_motion_mask_path = os.path.join(path, "synthetic_tof", f"{fid:04d}.npy")
        # if os.path.exists(os.path.join(path, "dynamic_ROI_mask")):
        #     tof_motion_mask = np.load(tof_motion_mask_path).astype(np.float32) / 255.0

        cam_infos.append(CameraInfo(
            uid=fid, frame_id=fid, 
            # Color
            R=R, T=T, FovY=FovY, FovX=FovX, 
            fx=fx*args.color_scale_factor, fy=fy*args.color_scale_factor, cx=cx*args.color_scale_factor, cy=cy*args.color_scale_factor,
            image_name=color_image_name, image=color_image, image_path=color_image_path,
            width=args.color_image_width*args.color_scale_factor, height=args.color_image_height*args.color_scale_factor,
            # Time-of-Flight
            R_tof=R_tof, T_tof=T_tof, FovY_tof=FovY_tof, FovX_tof=FovX_tof, 
            fx_tof=fx_tof*args.tof_scale_factor, fy_tof=fy_tof*args.tof_scale_factor, cx_tof=cx_tof*args.tof_scale_factor, cy_tof=cy_tof*args.tof_scale_factor,
            tof_image_name=tof_image_name, tof_image=tof_image, tof_image_path=tof_image_path,
            distance_image_name=distance_image_name, distance_image=distance_image, distance_image_path=distance_image_path,
            tof_width=args.tof_image_width*args.tof_scale_factor, tof_height=args.tof_image_height*args.tof_scale_factor,
            # ToF Raw Quads
            tofQuad0_im=tofQuad_ims[0], tofQuad1_im=tofQuad_ims[1], tofQuad2_im=tofQuad_ims[2], tofQuad3_im=tofQuad_ims[3],
            dc_offset=dc_offset,
            forward_flow=forward_flow, backward_flow=backward_flow,
            # Others
            znear=znear, zfar=zfar, depth_range=depth_range, phase_offset=phase_offset))
    return cam_infos

def readFToRFSceneInfo(path, args):
    # Load cameras
    tof_intrinsics, tof_extrinsics = get_camera_params(
        os.path.join(path, 'cams', f'tof_intrinsics.npy'), 
        os.path.join(path, 'cams', 'tof_extrinsics.npy'), 
        args.total_num_views, ftorf=True)
    color_intrinsics, color_extrinsics = get_camera_params(
        os.path.join(path, 'cams', f'color_intrinsics.npy'), 
        os.path.join(path, 'cams', 'color_extrinsics.npy'), 
        args.total_num_views, ftorf=True)

    phase_offset_path = os.path.join(path, 'cams', 'phase_offset.npy')
    if args.phase_offset != -99.0:
        phase_offset = np.array(args.phase_offset).astype(np.float32)
    elif os.path.exists(phase_offset_path):
        phase_offset = np.load(phase_offset_path).astype(np.float32)
    else:
        phase_offset = np.array(0.0).astype(np.float32)
    depth_range_path = os.path.join(path, 'cams', 'depth_range.npy')
    if os.path.exists(depth_range_path):
        depth_range = np.load(depth_range_path).astype(np.float32)
    else:
        depth_range = np.array(args.depth_range).astype(np.float32)
    dc_offset_path = os.path.join(path, 'cams', 'dc_offset.npy')
    if os.path.exists(dc_offset_path):
        dc_offset = np.load(dc_offset_path).astype(np.float32)
    else:
        dc_offset = np.array(args.dc_offset).astype(np.float32)
    quad_values_scale_factor_path = os.path.join(path, 'cams', 'quad_values_scale_factor.npy')
    if args.quad_scale != -1.0:
        quad_values_scale_factor = np.array(args.quad_scale).astype(np.float32)
    elif os.path.exists(quad_values_scale_factor_path):
        quad_values_scale_factor = np.load(quad_values_scale_factor_path).astype(np.float32)
    else:
        quad_values_scale_factor = np.array(1.0).astype(np.float32)
    znear = args.min_depth_fac * depth_range * 0.9
    zfar = args.max_depth_fac * depth_range * 1.1

    if args.tof_permutation != "":
        tof_permutation = np.array([int(i) for i in args.tof_permutation.split(",")])
    elif os.path.exists(os.path.join(path, 'tof_permutation.npy')):
        tof_permutation = np.load(os.path.join(path, 'tof_permutation.npy'))
    else:
        tof_permutation = np.array([0, 1, 2, 3])
    
    # Create splits
    cam_infos_unsorted = readFToRFCameras(path, tof_extrinsics, tof_intrinsics, color_extrinsics, color_intrinsics, depth_range, phase_offset, dc_offset, znear, zfar, quad_values_scale_factor, args)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    train_cam_infos = cam_infos
    test_cam_infos = cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if nerf_normalization["radius"] == 0.0:
        nerf_normalization["radius"] = 1.0
    nerf_normalization['scene_scale'] = depth_range.item() * 0.55
    nerf_normalization['tof_permutation'] = tof_permutation.tolist()
    nerf_normalization['tof_inverse_permutation'] = np.argsort(tof_permutation).tolist()

    # Init pcd
    ply_path = os.path.join(args.model_path, "points3d.ply")
    if args.init_method == "random":
        min_bounds, max_bounds = calculateSceneBounds(train_cam_infos, args)
        total_num_pts = args.num_points
        print(f"Generating random point cloud ({total_num_pts})...")

        # Init xyz
        xyz_all = np.random.uniform(min_bounds, max_bounds, (total_num_pts, 3))
        
        # Init phasor
        shs_phase = PA2SH(np.random.random((total_num_pts, 1)) * 2.0 * np.pi)
        shs_amp = PA2SH(np.ones((total_num_pts, 1)) * args.initial_amplitude)
        phases = SH2PA(shs_phase)
        amplitudes = SH2PA(shs_amp)
    elif args.init_method == "phase":
        cam_fid_list = [idx for idx in range(len(train_cam_infos[::4]))]
        
        xyz_all, shs_amp_all, shs_color_all = [], [], []
        for fid in cam_fid_list:
            tof_cam = train_cam_infos[fid]
            tof_depth_height = math.ceil(tof_cam.tof_height / args.phase_resolution_stride)
            tof_depth_width = math.ceil(tof_cam.tof_width / args.phase_resolution_stride)

            # Pixel space
            xy_screen = np.indices((tof_depth_height, tof_depth_width)).transpose(1, 2, 0).reshape(-1, 2).astype(np.float32)[:, ::-1] * args.phase_resolution_stride
            xy_screen = xy_screen.astype(np.int16)

            num_pts = xy_screen.shape[0]
            xyzw = np.empty((num_pts, 4))
            view_mat = getWorld2View2(tof_cam.R_tof, tof_cam.T_tof)

            # Normalize to [-WInMeters/2, WInMeters/2] and [-HInMeters/2, HInMeters/2]
            WInMeters = tof_cam.znear * np.tan(tof_cam.FovX_tof / 2.0) * 2.0
            HInMeters = tof_cam.znear * np.tan(tof_cam.FovY_tof / 2.0) * 2.0
            xyzw[:, 0] = (xy_screen[:, 0] * 2.0 / tof_cam.tof_width - 1.0) * WInMeters / 2.0
            xyzw[:, 1] = (xy_screen[:, 1] * 2.0 / tof_cam.tof_height - 1.0) * HInMeters / 2.0

            # Distances to Light
            z = depth_from_tof(tof_cam.tof_image[xy_screen[:, 1], xy_screen[:, 0], :], depth_range, phase_offset).reshape(num_pts, 1) 
            z_2 = z + depth_range / 2.0

            # Hardcoded for unwrapping ftorf scenes because camera is fixed 
            z_hardcoded_ = [
                [z_i for z_i in (z[i], z_2[i]) if znear < z_i <= 10.5]
                for i in range(num_pts)
            ]
            z_hardcoded = []
            for i in range(num_pts):
                h_, w_ = i // tof_depth_width, i % tof_depth_height
                if len(z_hardcoded_[i]) == 2: 
                    if tof_cam.tof_image[h_, w_, 2] < 0.04:
                        z_hardcoded.append(z_hardcoded_[i][1])
                    else:
                        z_hardcoded.append(z_hardcoded_[i][0])
                else:
                    z_hardcoded.append(z_hardcoded_[i][0])
            z_hardcoded = np.array(z_hardcoded).reshape(num_pts, 1)

            # Camera space.
            dists2pixInMeters = np.sqrt(np.square(xyzw[:, 0]) + np.square(xyzw[:, 1]) + np.square(tof_cam.znear))
            np.true_divide(xyzw[:, 0], dists2pixInMeters, out=xyzw[:, 0]) 
            np.true_divide(xyzw[:, 1], dists2pixInMeters, out=xyzw[:, 1])
            np.multiply(xyzw[:, 0:1], z_hardcoded, out=xyzw[:, 0:1])
            np.multiply(xyzw[:, 1:2], z_hardcoded, out=xyzw[:, 1:2])
            xyzw[:, 2:3] = np.sqrt(np.square(z_hardcoded) - np.square(xyzw[:, 0:1]) - np.square(xyzw[:, 1:2]))
            xyzw[:, 3:4] = np.ones((num_pts, 1))

            # World space.
            xyz = (np.linalg.inv(view_mat) @ xyzw.T).T[:, :3]
            shs_color = RGB2SH(tof_cam.tof_image[xy_screen[:, 1], xy_screen[:, 0], 2].reshape(-1, 1) * np.ones((1, 3), dtype=np.float32))
            shs_amp = PA2SH(tof_cam.tof_image[xy_screen[:, 1], xy_screen[:, 0], 2].reshape(-1, 1) * np.square(z_hardcoded))
            xyz_all.append(xyz), shs_amp_all.append(shs_amp), shs_color_all.append(shs_color)
            break

        xyz_all = np.concatenate(xyz_all, axis=0)
        shs_amp_all = np.concatenate(shs_amp_all, axis=0)
        shs_color_all = np.concatenate(shs_color_all, axis=0)

        total_num_pts = xyz_all.shape[0]
        print(f"Generating point cloud based on depth from the canonical frame ({total_num_pts})...")

        shs_phase = PA2SH(np.zeros((total_num_pts, 1)).astype(np.float32))
        phases = SH2PA(shs_phase) 
        amplitudes = SH2PA(shs_amp_all)

    if args.init_static_dynamic_separation:
        xyz_all = np.concatenate([xyz_all, np.random.uniform(min_bounds, max_bounds, (total_num_pts, 3))], axis=0)
        phases = np.concatenate([phases, phases], axis=0)
        amplitudes = np.concatenate([amplitudes, amplitudes], axis=0)

        seg_colors_static = np.repeat(np.array([[0.0, 0.0, 1.0]]), total_num_pts, axis=0) # Static points
        seg_colors_dynamic = np.repeat(np.array([[1.0, 0.0, 0.0]]), total_num_pts, axis=0) # Dynamic points
        seg_colors = np.concatenate([seg_colors_static, seg_colors_dynamic], axis=0)
    else:
        seg_colors = np.repeat(np.array([[1.0, 0.0, 0.0]]), xyz_all.shape[0], axis=0) # All dynamic

    colors = SH2RGB(RGB2SH(np.random.random((xyz_all.shape[0], 3))))
    pcd = BasicPointCloud(points=xyz_all, colors=seg_colors, normals=np.zeros_like(xyz_all), phases=phases, amplitudes=amplitudes, seg_colors=seg_colors)
    
    colors *= 255.0
    seg_colors *= 255.0
    storePly(ply_path, xyz_all, seg_colors, phases, amplitudes, seg_colors)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd, 
                           train_cameras=train_cam_infos, 
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           spiral_cameras=[])
    return scene_info

def readFToRFDepthMaps(path, model_path, iteration, args):
    # Load cameras
    tof_intrinsics, tof_extrinsics = get_camera_params(
        os.path.join(path, 'cams', f'tof_intrinsics.npy'), 
        os.path.join(path, 'cams', 'tof_extrinsics.npy'), 
        args.total_num_views)
    color_intrinsics, color_extrinsics = get_camera_params(
        os.path.join(path, 'cams', f'color_intrinsics.npy'), 
        os.path.join(path, 'cams', 'color_extrinsics.npy'), 
        args.total_num_views)
    
    phase_offset_path = os.path.join(path, 'cams', 'phase_offset.npy')
    if args.phase_offset != -99.0:
        phase_offset = np.array(args.phase_offset).astype(np.float32)
    elif os.path.exists(phase_offset_path):
        phase_offset = np.load(phase_offset_path).astype(np.float32)
    else:
        phase_offset = np.array(0.0).astype(np.float32)
    depth_range_path = os.path.join(path, 'cams', 'depth_range.npy')
    if os.path.exists(depth_range_path):
        depth_range = np.load(depth_range_path).astype(np.float32)
    else:
        depth_range = np.array(args.depth_range).astype(np.float32)
    dc_offset_path = os.path.join(path, 'cams', 'dc_offset.npy')
    if os.path.exists(dc_offset_path):
        dc_offset = np.load(dc_offset_path).astype(np.float32)
    else:
        dc_offset = np.array(args.dc_offset).astype(np.float32)
    quad_values_scale_factor_path = os.path.join(path, 'cams', 'quad_values_scale_factor.npy')
    if os.path.exists(quad_values_scale_factor_path):
        quad_values_scale_factor = np.load(quad_values_scale_factor_path).astype(np.float32)
    else:
        quad_values_scale_factor = np.array(1.0).astype(np.float32)
    znear = args.min_depth_fac * depth_range * 0.9
    zfar = args.max_depth_fac * depth_range * 1.1
    
    if args.tof_permutation != "":
        tof_permutation = np.array([int(i) for i in args.tof_permutation.split(",")])
    elif os.path.exists(os.path.join(path, 'tof_permutation.npy')):
        tof_permutation = np.load(os.path.join(path, 'tof_permutation.npy'))
    else:
        tof_permutation = np.array([0, 1, 2, 3])
    
    # Create splits
    cam_infos_unsorted = readFToRFCameras(path, tof_extrinsics, tof_intrinsics, color_extrinsics, color_intrinsics, depth_range, phase_offset, dc_offset, znear, zfar, quad_values_scale_factor, args)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    nerf_normalization = getNerfppNorm(cam_infos)
    if nerf_normalization["radius"] == 0.0:
        nerf_normalization["radius"] = 1.0
    nerf_normalization['scene_scale'] = depth_range.item() * 0.55
    nerf_normalization['tof_permutation'] = tof_permutation.tolist()
    nerf_normalization['tof_inverse_permutation'] = np.argsort(tof_permutation).tolist()

    rendered_depth_images = {}
    for fp in os.listdir(os.path.join(model_path, f"ours_{iteration}", "renders", "depth_norm_tof_cam")):
        if fp.endswith(".npy"):
            idx = int(fp.split(".")[0])
            depth_tof_cam_im = np.load(os.path.join(model_path, f"ours_{iteration}", "renders", "depth_norm_tof_cam", fp))
            rendered_depth_images[idx] = depth_tof_cam_im

    # Initialize point clouds
    for fid in range(len(cam_infos)):
        os.makedirs(os.path.join(model_path, "proxy_pcd", f"frame_{fid}", "point_cloud", f"iteration_{iteration}"), exist_ok=True)
        ply_path = os.path.join(model_path, "proxy_pcd", f"frame_{fid}", f"input.ply")

        tof_cam = cam_infos[fid]
        tof_depth_height = math.ceil(tof_cam.tof_height)
        tof_depth_width = math.ceil(tof_cam.tof_width)

        # Pixel space
        xy_screen = np.indices((tof_depth_height, tof_depth_width)).transpose(1, 2, 0).reshape(-1, 2).astype(np.float32)[:, ::-1]
        xy_screen = xy_screen.astype(np.int16)

        num_pts = xy_screen.shape[0]
        xyzw = np.empty((num_pts, 4))
        view_mat = getWorld2View2(tof_cam.R_tof, tof_cam.T_tof)

        # Normalize to [-WInMeters/2, WInMeters/2] and [-HInMeters/2, HInMeters/2]
        WInMeters = tof_cam.znear * np.tan(tof_cam.FovX_tof / 2.0) * 2.0
        HInMeters = tof_cam.znear * np.tan(tof_cam.FovY_tof / 2.0) * 2.0
        xyzw[:, 0] = (xy_screen[:, 0] * 2.0 / tof_cam.tof_width - 1.0) * WInMeters / 2.0
        xyzw[:, 1] = (xy_screen[:, 1] * 2.0 / tof_cam.tof_height - 1.0) * HInMeters / 2.0
        xyzw = np.concatenate([xyzw, xyzw], axis=0)

        # Distances to Light
        ## input depth
        z_i = depth_from_tof(tof_cam.tof_image[xy_screen[:, 1], xy_screen[:, 0], :], depth_range, phase_offset).reshape(tof_depth_height, tof_depth_width).reshape(num_pts, 1)
        ## rendered depth
        z_r = rendered_depth_images[fid][xy_screen[:, 1], xy_screen[:, 0]].reshape(num_pts, 1) 
        ## Concatenate
        z = np.concatenate([z_i, z_r], axis=0)

        # Camera space.
        dists2pixInMeters = np.sqrt(np.square(xyzw[:, 0]) + np.square(xyzw[:, 1]) + np.square(tof_cam.znear))
        np.true_divide(xyzw[:, 0], dists2pixInMeters, out=xyzw[:, 0]) 
        np.true_divide(xyzw[:, 1], dists2pixInMeters, out=xyzw[:, 1])
        np.multiply(xyzw[:, 0:1], z, out=xyzw[:, 0:1])
        np.multiply(xyzw[:, 1:2], z, out=xyzw[:, 1:2])
        xyzw[:, 2:3] = np.sqrt(np.square(z) - np.square(xyzw[:, 0:1]) - np.square(xyzw[:, 1:2]))
        xyzw[:, 3:4] = 1

        # World space.
        xyz = (np.linalg.inv(view_mat) @ xyzw.T).T[:, :3]
        colors = np.tile([1, 0, 0], (xyz.shape[0], 1)).astype(np.float32)
        colors[xyz.shape[0]//2:, :] = np.tile([0, 0, 1], (xyz.shape[0]//2, 1)).astype(np.float32)

        phases = SH2PA(PA2SH(np.zeros((xyz.shape[0], 1)).astype(np.float32))) 
        amplitudes = SH2PA(PA2SH(np.zeros((xyz.shape[0], 1)).astype(np.float32))) 
        seg_colors = np.repeat(np.array([[0.0, 0.0, 0.0]]), xyz.shape[0], axis=0) # All dynamic
        
        colors *= 255.0
        seg_colors *= 255.0
        storePly(ply_path, xyz, colors, phases=phases, amplitudes=amplitudes, seg_colors=seg_colors)

    return len(cam_infos), cam_infos

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "ToRF": readToRFSceneInfo,
    "ToRF_proxy_pcd": readToRFDepthMaps,
    "FToRF": readFToRFSceneInfo,
    "FToRF_proxy_pcd": readFToRFDepthMaps,
}