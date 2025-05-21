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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        self.dynamic = args.dynamic

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            self.scene_type = "colmap"
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.bg_color, args.eval)
            self.scene_type = "blender"
        elif os.path.exists(os.path.join(args.source_path, "tofType0")):
            print("Found ToF quad folder, assuming ftorf dataset")
            scene_info = sceneLoadTypeCallbacks["FToRF"](args.source_path, args)
            self.scene_type = "ftorf"
        elif os.path.exists(os.path.join(args.source_path, "tof")):
            print("Found ToF phasors folder, assuming torf dataset")
            scene_info = sceneLoadTypeCallbacks["ToRF"](args.source_path, args.eval, args)
            self.scene_type = "torf"
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            json_cams_full = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.spiral_cameras:
                camlist.extend(scene_info.spiral_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
                json_cams_full.append(camera_to_JSON(id, cam, save_full=True))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file, indent=4)
            with open(os.path.join(self.model_path, "cameras_full.json"), 'w') as file:
                json.dump(json_cams_full, file, indent=4)
            with open(os.path.join(self.model_path, "nerf_normalization.json"), 'w') as file:
                json.dump(scene_info.nerf_normalization, file, indent=4)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.scene_extent = scene_info.nerf_normalization["scene_scale"] if "scene_scale" in scene_info.nerf_normalization.keys() else self.cameras_extent
        if "tof_permutation" in scene_info.nerf_normalization.keys():
            self.tof_permutation = scene_info.nerf_normalization["tof_permutation"]
            self.tof_inverse_permutation = scene_info.nerf_normalization["tof_inverse_permutation"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras)
            print("Loading Spiral Cameras")
            self.spiral_cameras = cameraList_from_camInfos(scene_info.spiral_cameras)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud_full.ply"))
            if self.scene_type == "ftorf":
                self.gaussians.load_phase_offset(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                    "phase_offset.npy"))
                self.gaussians.load_dc_offset(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                    "dc_offset.npy"))
            if self.dynamic:
                self.gaussians.deform_model.load(os.path.join(self.model_path,
                                                              "deform_model",
                                                              "iteration_" + str(self.loaded_iter),
                                                              "deform_model.pth"))
            self.gaussians.scene_extent = self.scene_extent
            self.gaussians.isotropic = args.isotropic_gaussians
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.scene_extent, args)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), sibr_only=True)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_full.ply"), sibr_only=False)
        self.gaussians.save_phase_offset(os.path.join(point_cloud_path, "phase_offset.npy"))
        self.gaussians.save_dc_offset(os.path.join(point_cloud_path, "dc_offset.npy"))
        
        if self.dynamic:
            deform_model_path = os.path.join(self.model_path, "deform_model/iteration_{}".format(iteration))
            self.gaussians.deform_model.save(os.path.join(deform_model_path, "deform_model.pth"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getSpiralCameras(self):
        return self.spiral_cameras

class Scene_tmp:

    def __init__(self, args : ModelParams, other_args):
        self.model_path = args.model_path

        num_frames, cam_infos = sceneLoadTypeCallbacks["FToRF_proxy_pcd"](args.source_path, args.model_path, other_args.iteration, args)
        self.num_frames = num_frames

        json_cams = []
        for id, cam in enumerate(cam_infos):
            json_cams.append(camera_to_JSON(id, cam))

        for fid in range(num_frames):
            with open(os.path.join(self.model_path, "proxy_pcd", f"frame_{fid}", "cameras.json"), 'w') as file:
                json.dump(json_cams, file, indent=4)

            source_file = os.path.join(self.model_path, "point_cloud", f"iteration_{other_args.iteration}", "point_cloud.ply")
            destination_folder = os.path.join(self.model_path, "proxy_pcd", f"frame_{fid}", "point_cloud", f"iteration_{other_args.iteration}")
            os.system(f'copy "{source_file}" "{destination_folder}\\"')
