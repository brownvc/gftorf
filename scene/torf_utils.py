import os
import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
from scene.cameras import ToFCamera

# Image utils
def to8b(x):
    return (255*np.clip(x, 0, 1)).astype(np.uint8)

def normalize_im_max(im):
    if np.max(im) == 0.0:
        return im
    im = im / np.max(np.abs(im))
    im[np.isnan(im)] = 0.
    return im

def normalize_im(im):
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    im[np.isnan(im)] = 0.
    return np.clip(im, 0, 1)

def normalize_im_gt(im, im_gt):
    im = (im - np.min(im_gt)) / (np.max(im_gt) - np.min(im_gt))
    im[np.isnan(im)] = 0.
    return np.clip(im, 0, 1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def resize_all_images(images, width, height, method=cv2.INTER_AREA):
    resized_images = []
    for i in range(images.shape[0]):
        resized_images.append(cv2.resize(images[i], (width, height), interpolation=method))
    return np.stack(resized_images, axis=0)

def scale_image(image, scale=1, interpolation=cv2.INTER_AREA):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)

def tof_to_image(tof):
    cmap = plt.get_cmap('seismic')
    # Remove outliers
    tof = np.clip(tof, -0.5, 0.5)
    norm = plt.Normalize(-0.5, 0.5)
    tof_rgb = cmap(norm(tof))[:, :, :3]
    
    return (255 * tof_rgb).astype(np.uint8)

# ToF/Depth utils
def depth_from_tof(tof, depth_range, phase_offset=0.0):
    tof_phase = np.arctan2(tof[..., 1:2], tof[..., 0:1])
    tof_phase -= phase_offset
    tof_phase[tof_phase < 0] = tof_phase[tof_phase < 0] + 2 * np.pi
    return tof_phase * depth_range / (4 * np.pi)

def depth_from_tof_torch(tof, depth_range, phase_offset=0.0):
    real = torch.where(torch.abs(tof[0]) < 1e-6, torch.full_like(tof[0], 1e-6), tof[0])
    tof_phase = torch.atan2(tof[1], real)
    tof_phase -= phase_offset
    tof_phase = torch.where(tof_phase < 0, tof_phase + 2 * torch.pi, tof_phase)
    return tof_phase * depth_range / (4 * torch.pi)

def tof_from_depth(depth, amp, depth_range):
    tof_phase = depth * 4 * np.pi / depth_range
    amp *= 1. / np.maximum(depth * depth, (depth_range * 0.1) * (depth_range * 0.1))
    return np.stack([np.cos(tof_phase) * amp, np.sin(tof_phase) * amp, amp], axis=-1)

def z_depth_to_distance_map(z_depth, K):
    x, y = np.meshgrid(np.arange(z_depth.shape[1]), np.arange(z_depth.shape[0]))
    return np.sqrt(((x - K[0, 2]) * z_depth / K[0, 0]) ** 2 + ((y - K[1, 2]) * z_depth / K[1, 1]) ** 2 + z_depth ** 2)

def distance_to_z_depth(distance_map, K):
    x, y = np.meshgrid(np.arange(distance_map.shape[1]), np.arange(distance_map.shape[0]))
    return distance_map / np.sqrt(((x - K[0, 2]) / K[0, 0]) ** 2 + ((y - K[1, 2]) / K[1, 1]) ** 2 + 1)

# Flow utils
def distance_to_points3d(distance_map, viewpoint_cam : ToFCamera):
    H, W = distance_map.shape[1:]
    fx, fy, cx, cy = viewpoint_cam.fx, viewpoint_cam.fy, viewpoint_cam.cx, viewpoint_cam.cy
    u, v = torch.meshgrid(torch.arange(W, device=distance_map.device), torch.arange(H, device=distance_map.device), indexing='xy')

    z_cam = distance_map / torch.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1)
    x_cam = (u - cx) * z_cam / fx
    y_cam = (v - cy) * z_cam / fy
    points3d_cam = torch.cat([x_cam, y_cam, z_cam], dim=0)

    points3d_cam_flat = points3d_cam.view(3, -1)
    points3d_cam_homog = torch.cat([points3d_cam_flat, torch.ones((1, H * W), device=points3d_cam_flat.device)], dim=0)
    points3d_world = (torch.inverse(viewpoint_cam.world_view_transform) @ points3d_cam_homog)[:3, :].reshape(3, H, W)
    return points3d_world

def project_points_to_cam(points3d, viewpoint_cam : ToFCamera):
    points3d_homog = torch.cat([points3d, torch.ones((points3d.shape[0], 1), device=points3d.device)], dim=1)
    points3d_cam = (viewpoint_cam.world_view_transform_tof.transpose(1,0) @ points3d_homog.T)[:3, :].T
    return points3d_cam

def project_points(points3d, viewpoint_cam : ToFCamera):
    H, W = points3d.shape[1:]
    points3d_flat = points3d.view(3, -1)
    points3d_homog = torch.cat([points3d_flat, torch.ones((1, H * W), device=points3d_flat.device)], dim=0)
    points2d_homog = (viewpoint_cam.K_tof @ (viewpoint_cam.world_view_transform_tof.transpose(1,0) @ points3d_homog)[:3, :])
    points2d = (points2d_homog[:2, :] / (points2d_homog[2:, :] + 1e-7)).reshape(2, H, W)
    return points2d

def project_points_subset(points3d, viewpoint_cam : ToFCamera):
    N = points3d.shape[0]
    points3d_T = points3d.T
    points3d_homog = torch.cat([points3d_T, torch.ones((1, N), device=points3d_T.device)], dim=0)
    points2d_homog = (viewpoint_cam.K_tof @ (viewpoint_cam.world_view_transform_tof.transpose(1,0) @ points3d_homog)[:3, :])
    points2d = (points2d_homog[:2, :] / (points2d_homog[2:, :] + 1e-7)).reshape(2, N).T
    return points2d

def project_flow(points2d_curr, points3d_curr, flow3d, viewpoint_cam : ToFCamera):
    H, W = points3d_curr.shape[1:]
    points3d_next_flat = points3d_curr.view(3, -1) + flow3d.view(3, -1)
    points3d_next_homog = torch.cat([points3d_next_flat, torch.ones((1, H * W), device=points3d_next_flat.device)], dim=0)
    points2d_next_homog = (viewpoint_cam.K_tof @ (viewpoint_cam.world_view_transform_tof.transpose(1,0) @ points3d_next_homog)[:3, :])
    points2d_next = (points2d_next_homog[:2, :] / (points2d_next_homog[2:, :] + 1e-7)).reshape(2, H, W)
    # points2d_next = torch.clamp(points2d_next, min=-max(H,W), max=max(H,W))
    # points2d_curr = torch.clamp(points2d_curr, min=-max(H,W), max=max(H,W))
    return points2d_next - points2d_curr

def visualize_2d_flow_arrows(flow2d, save_path, step=5):
    h, w = flow2d.shape[:2]
    y, x = np.mgrid[0:h:step, 0:w:step]
    fx, fy = flow2d[::step, ::step, 0], -flow2d[::step, ::step, 1]
    
    plt.figure(figsize=(8, 8))
    plt.ioff()
    plt.imshow(np.zeros((h, w)), cmap="gray")
    plt.quiver(x, y, fx, fy, color='r')
    plt.savefig(save_path)
    plt.close()

# Deprecated
def save_flow_images(model_path, iteration, projected_flow3d, target_flow2d, flow_type="forward"):
    projected_flow3d_np = projected_flow3d.cpu().detach().numpy().transpose(1, 2, 0)
    target_flow2d_np = target_flow2d.cpu().detach().numpy().transpose(1, 2, 0)
    projected_flow2d_np = projected_flow3d_np[:, :, :2]
    error_flow2d_np = target_flow2d_np - projected_flow2d_np

    visualize_2d_flow_arrows(target_flow2d_np, os.path.join(model_path, f"tmp_debug_{flow_type}_flow2d_gt", f"{iteration:05d}.png"))
    visualize_2d_flow_arrows(projected_flow2d_np, os.path.join(model_path, f"tmp_debug_{flow_type}_flow2d", f"{iteration:05d}.png"))
    visualize_2d_flow_arrows(error_flow2d_np, os.path.join(model_path, f"tmp_debug_{flow_type}_flow2d_error", f"{iteration:05d}.png"))
    imageio.imwrite(os.path.join(model_path, f"tmp_debug_{flow_type}_flow3d", f"{iteration:05d}.png"), to8b(normalize_im(projected_flow3d_np)))

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow, gt_flows=None, display=False):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    if type(flow) != np.ndarray:
        flow = np.array(flow)

    UNKNOWN_FLOW_THRESH = 200
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)

    if gt_flows is not None:
        if isinstance(gt_flows, list):
            gt_flows = np.stack(gt_flows, axis=0)
        gt_u = gt_flows[..., 0]
        gt_v = gt_flows[..., 1]
        
        idxUnknow_gt = (abs(gt_u) > UNKNOWN_FLOW_THRESH) | (abs(gt_v) > UNKNOWN_FLOW_THRESH)
        gt_u[idxUnknow_gt] = 0
        gt_v[idxUnknow_gt] = 0

        # Dataloader loads missing flow files as np.nan arrays, which effects max calculation
        nan_gt = np.isnan(gt_u) | np.isnan(gt_v)
        gt_u[nan_gt] = 0
        gt_v[nan_gt] = 0

        gt_rad = np.sqrt(gt_u**2 + gt_v**2)

        maxrad = np.max(gt_rad)
    else:
        maxrad = max(-1, np.max(rad))

    if display:
        print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

    maxrad *= 1.1
    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

# Camera utils
def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       raise ValueError("Cannot normalize a zero vector")
    return v / norm

def get_camera_params(intrinsics_file, extrinsics_file, total_num_views=60, ftorf=False):
    if '.mat' in intrinsics_file:
        K = scipy.io.loadmat(intrinsics_file)['K']
    else:
        K = np.load(intrinsics_file)
    Ks = [np.copy(K) for _ in range(total_num_views)]

    if ftorf:
        exts = np.identity(4).astype(np.float32)[None, ...].repeat(total_num_views,0)
    else:
        exts = np.load(extrinsics_file)
    return Ks, exts

def normalize(v, axis=-1, epsilon=1e-6):
    norm = np.linalg.norm(v, ord=2, axis=axis, keepdims=True)
    return v / (norm + epsilon)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses): # c2w
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    hwf = c2w[:,4:5]
    rads = np.array(list(rads) + [1.])
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.sin(-theta), np.cos(-theta), np.sin(-theta * zrate), 1.]) * rads)
        z = normalize(-c + np.dot(c2w[:3,:4], np.array([0, 0, focal, 1.])))
        pose = np.eye(4)
        pose[:3, :4] = viewmatrix(z, up, c)
        render_poses.append(pose)

    return render_poses

def get_render_poses_spiral(focal_length, bounds_data, poses, N_views=60, N_rots=2):
    poses = np.array(poses)

    ## Focus distance
    if focal_length < 0:
        close_depth, inf_depth = bounds_data.min() * .9, bounds_data.max() * 5.
        dt = .75
        mean_dz = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)
        focal_length = mean_dz

    # Get average pose
    c2w = poses_avg(poses)
    c2w_path = c2w
    up = normalize(poses[:, :3, 1].sum(0))

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = bounds_data.min() * .2
    tt = (poses[:, :3, 3] - c2w[:3, 3])
    if np.sum(tt) < 1e-10:
        tt = np.array([1.0, 1.0, 1.0])

    rads = np.percentile(np.abs(tt), 90, 0) * np.array([1.0, 1.0, 1.0]) / 3.0

    # Generate poses for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal_length, zdelta, zrate=.5, rots=N_rots, N=N_views)
    render_poses = np.array(render_poses).astype(np.float32)

    return render_poses

def recenter_poses(poses):
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses, np.linalg.inv(c2w)

def cameraFrustumCorners(cam_info):
    """
    Calculate the world-space positions of the corners of the camera's view frustum.
    """
    aspect_ratio = cam_info.tof_width / cam_info.tof_height
    hnear = 2 * np.tan(cam_info.FovY_tof / 2) * cam_info.znear
    wnear = hnear * aspect_ratio
    hfar = 2 * np.tan(cam_info.FovX_tof / 2) * cam_info.zfar
    wfar = hfar * aspect_ratio

    # Camera's forward direction (z forward y down in SfM convention)
    forward = normalize_vector(np.linalg.inv(np.transpose(cam_info.R_tof))[:, 2])
    right = normalize_vector(np.linalg.inv(np.transpose(cam_info.R_tof))[:, 0])
    up = -normalize_vector(np.linalg.inv(np.transpose(cam_info.R_tof))[:, 1])

    # Camera position
    cam_pos = -np.linalg.inv(np.transpose(cam_info.R_tof)) @ cam_info.T_tof

    # Near plane corners
    nc_tl = cam_pos + forward * cam_info.znear + up * (hnear / 2) - right * (wnear / 2)
    nc_tr = cam_pos + forward * cam_info.znear + up * (hnear / 2) + right * (wnear / 2)
    nc_bl = cam_pos + forward * cam_info.znear - up * (hnear / 2) - right * (wnear / 2)
    nc_br = cam_pos + forward * cam_info.znear - up * (hnear / 2) + right * (wnear / 2)

    # Far plane corners
    fc_tl = cam_pos + forward * cam_info.zfar + up * (hfar / 2) - right * (wfar / 2)
    fc_tr = cam_pos + forward * cam_info.zfar + up * (hfar / 2) + right * (wfar / 2)
    fc_bl = cam_pos + forward * cam_info.zfar - up * (hfar / 2) - right * (wfar / 2)
    fc_br = cam_pos + forward * cam_info.zfar - up * (hfar / 2) + right * (wfar / 2)

    return np.array([nc_tl, nc_tr, nc_bl, nc_br, fc_tl, fc_tr, fc_bl, fc_br])

def calculateSceneBounds(cam_infos, args):
    cam_xyzs = np.array([-np.linalg.inv(np.transpose(cam_info.R_tof)) @ cam_info.T_tof for cam_info in cam_infos])
    cam_dirs = np.array([normalize_vector(np.linalg.inv(np.transpose(cam_info.R_tof))[:, 2]) for cam_info in cam_infos]) # SfM convention
    
    all_corners = []
    for cam_info in cam_infos:
        corners = cameraFrustumCorners(cam_info)
        all_corners.append(corners)
    
    # if args.debug:
    plt.ioff()
    # Visualize camera positions
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(cam_xyzs[:, 0], cam_xyzs[:, 1], cam_xyzs[:, 2], color = "green")

    # Visualize camera viewing directions
    for i in range(cam_dirs.shape[0]):
        view_dir = cam_dirs[i]
        scale = 0.05
        ax.quiver(cam_xyzs[i, 0], cam_xyzs[i, 1], cam_xyzs[i, 2], view_dir[0]*scale, view_dir[1]*scale, view_dir[2]*scale, color='red', length=3, normalize=True)

    # Visualize camera corners (to determine scene bounds)
    for i, cs in enumerate(all_corners):
        ax.scatter3D(cs[:, 0], cs[:, 1], cs[:, 2], color = "blue")
    plt.title("Camera Poses")
    # plt.legend()
    plt.savefig(os.path.join(args.model_path, "scene_bounds.png"))
    # plt.show()
    plt.close()

    all_corners = np.vstack(all_corners)
    min_bounds = np.min(all_corners, axis=0)
    max_bounds = np.max(all_corners, axis=0)

    return min_bounds, max_bounds

def compute_bounds(scene):
    # Paper depth map results. To match FToRF videos.
    synthetic_scenes = {
        "sliding_cube": (0.07, 0.24, 15),
        "occlusion": (0.03, 0.21, 15),
        "speed_test_texture": (0.08, 0.32, 15),
        "speed_test_chair": (0.08, 0.32, 15),
        "arcing_cube": (0.03, 0.38, 15),
        "z_motion_speed_test": (0.06, 0.34, 15),
        "acute_z_speed_test": (0.01, 0.52, 15),
    }

    has_gt = False
    if scene in synthetic_scenes:
        near_factor, far_factor, max_depth = synthetic_scenes[scene]
        near, far = near_factor * max_depth * 0.9, far_factor * max_depth * 1.1
        has_gt = True
    elif 'data_color' in scene:
        near, far = 0.45, 6.05
    else:
        near, far = 0.135, 10.725
    return near, far, has_gt