import sapien.core as sapien
import numpy as np
import pdb
from PIL import Image, ImageColor
import open3d as o3d
import json
import transforms3d as t3d
import cv2
import torch
import yaml
import trimesh
import math
from .._GLOBAL_CONFIGS import CONFIGS_PATH
import os
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig

try:
    import pytorch3d.ops as torch3d_ops

    def fps(points, num_points=1024, use_cuda=True):
        K = [num_points]
        if use_cuda:
            points = torch.from_numpy(points).cuda()
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.cpu().numpy()
            indices = indices.squeeze(0).cpu().numpy()
        else:
            points = torch.from_numpy(points)
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.numpy()
            indices = indices.squeeze(0).numpy()

        return sampled_points, indices

except:
    print("missing pytorch3d")

    def fps(points, num_points=1024, use_cuda=True):
        print("fps error: missing pytorch3d")
        exit()


def depth_to_pointcloud_rgbd(depth, rgb_img, intrinsic, cam2world_gl, depth_max=2.0):
    """从RGB-D深度图生成密集点云（世界坐标系）- 与训练时完全一致
    
    🎯 关键：此函数与 process_rgb_pcd.py 中的 depth_to_pointcloud() 完全一致
    用于确保训练和推理使用相同的点云生成逻辑
    
    使用OpenGL坐标系约定（与Sapien一致）:
    - 相机坐标系：X右，Y上，Z向后（物体在前方时Z<0）
    - depth = -position[..., 2] * 1000（Sapien保存方式）
    
    参数:
        depth: 深度图 (H, W)，单位为毫米
        rgb_img: RGB格式图像 (H, W, 3)，范围0-255（注意：必须是RGB格式，不是BGR）
        intrinsic: 相机内参矩阵 (3, 3)
        cam2world_gl: 相机到世界的变换矩阵 (4, 4)，OpenGL坐标系
        depth_max: 最大深度阈值（米），用于过滤远处背景点（默认2.0米）
    
    返回:
        pointcloud_with_color: (N, 6) 数组，每行为 [x, y, z, r, g, b]
    """
    h, w = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # OpenGL相机坐标系：Z向后，物体在前方时Z<0
    # depth = -position_z * 1000，所以 position_z = -depth / 1000
    z_opengl = -depth / 1000.0  # 注意负号！
    # 保持原来的X坐标负号（经验证，去掉反而偏移更大）
    x = -(u - cx) * z_opengl / fx
    y = (v - cy) * z_opengl / fy
    
    # 齐次坐标
    points_cam_homo = np.stack([x, y, z_opengl, np.ones_like(x)], axis=-1).reshape(-1, 4)
    
    # 关键优化1：过滤无效深度点和远处背景点
    depth_in_meters = depth / 1000.0
    valid_mask_depth = (depth > 0) & (depth_in_meters < depth_max)
    
    # 关键优化2：过滤深度不连续的边缘点（物体边界噪声）
    # 使用Sobel算子计算梯度的模
    sobel_x = cv2.Sobel(depth_in_meters, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_in_meters, cv2.CV_64F, 0, 1, ksize=3)
    depth_diff = np.abs(sobel_x) + np.abs(sobel_y)
    edge_mask = depth_diff < 0.05  # 5cm梯度阈值
    
    # 合并mask
    valid_mask = (valid_mask_depth & edge_mask).flatten()
    points_cam_homo = points_cam_homo[valid_mask]
    
    # 转换到世界坐标系
    points_world_homo = (cam2world_gl @ points_cam_homo.T).T
    points_world = points_world_homo[:, :3]
    
    # 🎯 与训练时一致：完全不裁剪！
    # 保留所有通过深度有效性检查的点
    
    # 添加颜色信息
    rgb_reshaped = rgb_img.reshape(-1, 3)[valid_mask]
    
    pointcloud_with_color = np.concatenate([points_world, rgb_reshaped / 255.0], axis=1)
    
    return pointcloud_with_color


class Camera:

    def __init__(self, bias=0, random_head_camera_dis=0, **kwags):
        """ """
        self.pcd_crop = kwags.get("pcd_crop", False)
        self.pcd_down_sample_num = kwags.get("pcd_down_sample_num", 0)
        self.pcd_crop_bbox = kwags.get("bbox", [[-0.6, -0.35, 0.7401], [0.6, 0.35, 2]])
        self.pcd_crop_bbox[0][2] += bias
        self.table_z_bias = bias
        self.random_head_camera_dis = random_head_camera_dis

        self.static_camera_config = []
        self.head_camera_type = kwags["camera"].get("head_camera_type", "D435")
        self.wrist_camera_type = kwags["camera"].get("wrist_camera_type", "D435")

        self.collect_head_camera = kwags["camera"].get("collect_head_camera", True)
        self.collect_wrist_camera = kwags["camera"].get("collect_wrist_camera", True)

        # embodiment = kwags.get('embodiment')
        # embodiment_config_path = os.path.join(CONFIGS_PATH, '_embodiment_config.yml')
        # with open(embodiment_config_path, 'r', encoding='utf-8') as f:
        #     embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
        # robot_file = embodiment_types[embodiment]['file_path']
        # if robot_file is None:
        #     raise "No embodiment files"

        # robot_config_file = os.path.join(robot_file, 'config.yml')
        # with open(robot_config_file, 'r', encoding='utf-8') as f:
        #     embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        # TODO
        self.static_camera_info_list = kwags["left_embodiment_config"]["static_camera_list"]
        self.static_camera_num = len(self.static_camera_info_list)

    def load_camera(self, scene):
        """
        Add cameras and set camera parameters
            - Including four cameras: left, right, front, head.
        """
        near, far = 0.1, 100
        camera_config_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")

        assert os.path.isfile(camera_config_path), "task config file is missing"

        with open(camera_config_path, "r", encoding="utf-8") as f:
            camera_args = yaml.load(f.read(), Loader=yaml.FullLoader)

        # sensor_mount_actor = scene.create_actor_builder().build_kinematic()

        # camera_args = get_camera_config()
        def create_camera(camera_info, random_head_camera_dis=0):
            if camera_info["type"] not in camera_args.keys():
                raise ValueError(f"Camera type {camera_info['type']} not supported")

            camera_config = camera_args[camera_info["type"]]
            cam_pos = np.array(camera_info["position"])
            vector = np.random.randn(3)
            random_dir = vector / np.linalg.norm(vector)
            cam_pos = cam_pos + random_dir * np.random.uniform(low=0, high=random_head_camera_dis)
            cam_forward = np.array(camera_info["forward"]) / np.linalg.norm(np.array(camera_info["forward"]))
            cam_left = np.array(camera_info["left"]) / np.linalg.norm(np.array(camera_info["left"]))
            up = np.cross(cam_forward, cam_left)
            mat44 = np.eye(4)
            mat44[:3, :3] = np.stack([cam_forward, cam_left, up], axis=1)
            mat44[:3, 3] = cam_pos

            # ========================= sensor camera =========================
            # sensor_config = StereoDepthSensorConfig()
            # sensor_config.rgb_resolution = (camera_config['w'], camera_config['h'])

            camera = scene.add_camera(
                name=camera_info["name"],
                width=camera_config["w"],
                height=camera_config["h"],
                fovy=np.deg2rad(camera_config["fovy"]),
                near=near,
                far=far,
            )
            camera.entity.set_pose(sapien.Pose(mat44))

            # ========================= sensor camera =========================
            # sensor_camera = StereoDepthSensor(
            #     sensor_config,
            #     sensor_mount_actor,
            #     sapien.Pose(mat44)
            # )
            # camera.entity.set_pose(sapien.Pose(camera_info['position']))
            # return camera, sensor_camera, camera_config
            return camera, camera_config

        # ================================= wrist camera =================================
        if self.collect_wrist_camera:
            wrist_camera_config = camera_args[self.wrist_camera_type]
            self.left_camera = scene.add_camera(
                name="left_camera",
                width=wrist_camera_config["w"],
                height=wrist_camera_config["h"],
                fovy=np.deg2rad(wrist_camera_config["fovy"]),
                near=near,
                far=far,
            )

            self.right_camera = scene.add_camera(
                name="right_camera",
                width=wrist_camera_config["w"],
                height=wrist_camera_config["h"],
                fovy=np.deg2rad(wrist_camera_config["fovy"]),
                near=near,
                far=far,
            )

        # ================================= sensor camera =================================
        # sensor_config = StereoDepthSensorConfig()
        # sensor_config.rgb_resolution = (wrist_camera_config['w'], wrist_camera_config['h'])
        # self.left_sensor_camera = StereoDepthSensor(
        #     sensor_config,
        #     sensor_mount_actor,
        #     sapien.Pose([0,0,0],[1,0,0,0])
        # )

        # self.right_sensor_camera = StereoDepthSensor(
        #     sensor_config,
        #     sensor_mount_actor,
        #     sapien.Pose([0,0,0],[1,0,0,0])
        # )

        # ================================= static camera =================================
        self.head_camera_id = None
        self.static_camera_list = []
        # self.static_sensor_camera_list = []
        self.static_camera_name = []
        # static camera list
        for i, camera_info in enumerate(self.static_camera_info_list):
            if camera_info.get("forward") == None:
                camera_info["forward"] = (-1 * np.array(camera_info["position"])).tolist()
            if camera_info.get("left") == None:
                camera_info["left"] = [
                    -camera_info["forward"][1],
                    camera_info["forward"][0],
                ] + [0]

            if camera_info["name"] == "head_camera":
                if self.collect_head_camera:
                    self.head_camera_id = i
                    camera_info["type"] = self.head_camera_type
                    # camera, sensor_camera, camera_config = create_camera(camera_info)
                    camera, camera_config = create_camera(camera_info,
                                                          random_head_camera_dis=self.random_head_camera_dis)
                    self.static_camera_list.append(camera)
                    self.static_camera_name.append(camera_info["name"])
                    # self.static_sensor_camera_list.append(sensor_camera)
                    self.static_camera_config.append(camera_config)
                    # ================================= sensor camera =================================
                    # camera_config = get_camera_config(camera_info['type'])
                    # cam_pos = np.array(camera_info['position'])
                    # cam_forward = np.array(camera_info['forward']) / np.linalg.norm(np.array(camera_info['forward']))
                    # cam_left = np.array(camera_info['left']) / np.linalg.norm(np.array(camera_info['left']))
                    # up = np.cross(cam_forward, cam_left)
                    # mat44 = np.eye(4)
                    # mat44[:3, :3] = np.stack([cam_forward, cam_left, up], axis=1)
                    # mat44[:3, 3] = cam_pos
                    # sensor_config = StereoDepthSensorConfig()
                    # sensor_config.rgb_resolution = (camera_config['w'], camera_config['h'])

                    # self.head_sensor = StereoDepthSensor(
                    #     sensor_config,
                    #     sensor_mount_actor,
                    #     sapien.Pose(mat44)
                    # )
            else:
                # camera, sensor_camera, camera_config = create_camera(camera_info)
                camera, camera_config = create_camera(camera_info)
                self.static_camera_list.append(camera)
                self.static_camera_name.append(camera_info["name"])
                # self.static_sensor_camera_list.append(sensor_camera)
                self.static_camera_config.append(camera_config)

        # observer camera
        self.observer_camera = scene.add_camera(
            name="observer_camera",
            width=320,
            height=240,
            fovy=np.deg2rad(93),
            near=near,
            far=far,
        )
        observer_cam_pos = np.array([0.0, 0.23, 1.33])
        observer_cam_forward = np.array([0, -1, -1.02])
        # observer_cam_left = np.array([1,-1, 0])
        observer_cam_left = np.array([1, 0, 0])
        observer_up = np.cross(observer_cam_forward, observer_cam_left)
        observer_mat44 = np.eye(4)
        observer_mat44[:3, :3] = np.stack([observer_cam_forward, observer_cam_left, observer_up], axis=1)
        observer_mat44[:3, 3] = observer_cam_pos
        self.observer_camera.entity.set_pose(sapien.Pose(observer_mat44))

        # world pcd camera
        self.world_camera1 = scene.add_camera(
            name="world_camera1",
            width=640,
            height=480,
            fovy=np.deg2rad(50),
            near=near,
            far=far,
        )
        world_cam_pos = np.array([0.4, -0.4, 1.6])
        world_cam_forward = np.array([-1, 1, -1.4])
        world_cam_left = np.array([-1, -1, 0])
        world_cam_up = np.cross(world_cam_forward, world_cam_left)
        world_cam_mat44 = np.eye(4)
        world_cam_mat44[:3, :3] = np.stack([world_cam_forward, world_cam_left, world_cam_up], axis=1)
        world_cam_mat44[:3, 3] = world_cam_pos
        self.world_camera1.entity.set_pose(sapien.Pose(world_cam_mat44))

        self.world_camera2 = scene.add_camera(
            name="world_camera1",
            width=640,
            height=480,
            fovy=np.deg2rad(50),
            near=near,
            far=far,
        )
        world_cam_pos = np.array([-0.4, -0.4, 1.6])
        world_cam_forward = np.array([1, 1, -1.4])
        world_cam_left = np.array([-1, 1, 0])
        world_cam_up = np.cross(world_cam_forward, world_cam_left)
        world_cam_mat44 = np.eye(4)
        world_cam_mat44[:3, :3] = np.stack([world_cam_forward, world_cam_left, world_cam_up], axis=1)
        world_cam_mat44[:3, 3] = world_cam_pos
        self.world_camera2.entity.set_pose(sapien.Pose(world_cam_mat44))

    def update_picture(self):
        # camera
        if self.collect_wrist_camera:
            self.left_camera.take_picture()
            self.right_camera.take_picture()

        for camera in self.static_camera_list:
            camera.take_picture()

        # ================================= sensor camera =================================
        # self.head_sensor.take_picture()
        # self.head_sensor.compute_depth()

    def update_wrist_camera(self, left_pose, right_pose):
        """
        Update rendering to refresh the camera's RGBD information
        (rendering must be updated even when disabled, otherwise data cannot be collected).
        """
        if self.collect_wrist_camera:
            self.left_camera.entity.set_pose(left_pose)
            self.right_camera.entity.set_pose(right_pose)

    def get_config(self) -> dict:
        res = {}

        def _get_config(camera):
            camera_intrinsic_cv = camera.get_intrinsic_matrix()
            camera_extrinsic_cv = camera.get_extrinsic_matrix()
            camera_model_matrix = camera.get_model_matrix()
            return {
                "intrinsic_cv": camera_intrinsic_cv,
                "extrinsic_cv": camera_extrinsic_cv,
                "cam2world_gl": camera_model_matrix,
            }

        if self.collect_wrist_camera:
            res["left_camera"] = _get_config(self.left_camera)
            res["right_camera"] = _get_config(self.right_camera)

        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = _get_config(camera)
            else:
                res[camera_name] = _get_config(camera)
        # ================================= sensor camera =================================
        # res['head_sensor'] = res['head_camera']
        # print(res)
        return res

    def get_rgb(self) -> dict:
        rgba = self.get_rgba()
        rgb = {}
        for camera_name, camera_data in rgba.items():
            rgb[camera_name] = {}
            rgb[camera_name]["rgb"] = camera_data["rgba"][:, :, :3]  # Exclude alpha channel
        return rgb
    
    # Get Camera RGBA
    def get_rgba(self) -> dict:

        def _get_rgba(camera):
            camera_rgba = camera.get_picture("Color")
            camera_rgba_img = (camera_rgba * 255).clip(0, 255).astype("uint8")
            return camera_rgba_img

        # ================================= sensor camera =================================
        # def _get_sensor_rgba(sensor):
        #     camera_rgba = sensor.get_rgb()
        #     camera_rgba_img = (camera_rgba * 255).clip(0, 255).astype("uint8")[:,:,:3]
        #     return camera_rgba_img

        res = {}

        if self.collect_wrist_camera:
            res["left_camera"] = {}
            res["right_camera"] = {}
            res["left_camera"]["rgba"] = _get_rgba(self.left_camera)
            res["right_camera"]["rgba"] = _get_rgba(self.right_camera)

        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = {}
                    res[camera_name]["rgba"] = _get_rgba(camera)
            else:
                res[camera_name] = {}
                res[camera_name]["rgba"] = _get_rgba(camera)
        # ================================= sensor camera =================================
        # res['head_sensor']['rgb'] = _get_sensor_rgba(self.head_sensor)

        return res

    def get_observer_rgb(self) -> dict:
        self.observer_camera.take_picture()

        def _get_rgb(camera):
            camera_rgba = camera.get_picture("Color")
            camera_rgb_img = (camera_rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
            return camera_rgb_img

        return _get_rgb(self.observer_camera)

    # Get Camera Segmentation
    def get_segmentation(self, level="mesh") -> dict:

        def _get_segmentation(camera, level="mesh"):
            # visual_id is the unique id of each visual shape
            seg_labels = camera.get_picture("Segmentation")  # [H, W, 4]
            colormap = sorted(set(ImageColor.colormap.values()))
            color_palette = np.array([ImageColor.getrgb(color) for color in colormap], dtype=np.uint8)
            if level == "mesh":
                label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
            elif level == "actor":
                label0_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
            return color_palette[label0_image]

        res = {
            # 'left_camera':{},
            # 'right_camera':{}
        }

        if self.collect_wrist_camera:
            res["left_camera"] = {}
            res["right_camera"] = {}
            res["left_camera"][f"{level}_segmentation"] = _get_segmentation(self.left_camera, level=level)
            res["right_camera"][f"{level}_segmentation"] = _get_segmentation(self.right_camera, level=level)

        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = {}
                    res[camera_name][f"{level}_segmentation"] = _get_segmentation(camera, level=level)
            else:
                res[camera_name] = {}
                res[camera_name][f"{level}_segmentation"] = _get_segmentation(camera, level=level)
        return res

    # Get Camera Depth
    def get_depth(self) -> dict:

        def _get_depth(camera):
            position = camera.get_picture("Position")
            depth = -position[..., 2]
            depth_image = (depth * 1000.0).astype(np.float64)
            return depth_image

        def _get_sensor_depth(sensor):
            depth = sensor.get_depth()
            depth = (depth * 1000.0).astype(np.float64)
            return depth

        res = {}
        rgba = self.get_rgba()

        if self.collect_wrist_camera:
            res["left_camera"] = {}
            res["right_camera"] = {}
            res["left_camera"]["depth"] = _get_depth(self.left_camera)
            res["right_camera"]["depth"] = _get_depth(self.right_camera)
            res["left_camera"]["depth"] *= rgba["left_camera"]["rgba"][:, :, 3] / 255
            res["right_camera"]["depth"] *= rgba["right_camera"]["rgba"][:, :, 3] / 255
        
        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = {}
                    res[camera_name]["depth"] = _get_depth(camera)
                    res[camera_name]["depth"] *= rgba[camera_name]["rgba"][:, :, 3] / 255
            else:
                res[camera_name] = {}
                res[camera_name]["depth"] = _get_depth(camera)
                res[camera_name]["depth"] *= rgba[camera_name]["rgba"][:, :, 3] / 255
        # res['head_sensor']['depth'] = _get_sensor_depth(self.head_sensor)

        return res

    # Get World PointCloud
    def get_world_pcd(self):
        self.world_camera1.take_picture()
        self.world_camera2.take_picture()

        def _get_camera_pcd(camera, color=True):
            rgba = camera.get_picture_cuda("Color").torch()  # [H, W, 4]
            position = camera.get_picture_cuda("Position").torch()
            model_matrix = camera.get_model_matrix()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_matrix = torch.tensor(model_matrix, dtype=torch.float32).to(device)

            # Extract valid three-dimensional points and corresponding color data.
            valid_mask = position[..., 3] < 1
            points_opengl = position[..., :3][valid_mask]
            points_color = rgba[valid_mask][:, :3]
            # Transform into the world coordinate system.
            points_world = (torch.bmm(
                points_opengl.view(1, -1, 3),
                model_matrix[:3, :3].transpose(0, 1).view(-1, 3, 3),
            ).squeeze(1) + model_matrix[:3, 3])

            # Format color data.
            points_color = torch.clamp(points_color, 0, 1)
            points_world = points_world.squeeze(0)

            # Convert the tensor back to a NumPy array for use with Open3D.
            points_world_np = points_world.cpu().numpy()
            points_color_np = points_color.cpu().numpy()
            # print(points_world_np.shape, points_color_np.shape)

            res_pcd = (np.hstack((points_world_np, points_color_np)) if color else points_world_np)
            return res_pcd

        pcd1 = _get_camera_pcd(self.world_camera1, color=True)
        pcd2 = _get_camera_pcd(self.world_camera2, color=True)
        res_pcd = np.vstack((pcd1, pcd2))

        return res_pcd
        pcd_array, index = fps(res_pcd[:, :3], 2000)
        index = index.detach().cpu().numpy()[0]

        return pcd_array

    # Get Camera PointCloud
    def get_pcd(self, if_combine=False):

        def _get_camera_pcd(camera, point_num=0):
            rgba = camera.get_picture_cuda("Color").torch()  # [H, W, 4]
            position = camera.get_picture_cuda("Position").torch()
            model_matrix = camera.get_model_matrix()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_matrix = torch.tensor(model_matrix, dtype=torch.float32).to(device)

            # Extract valid three-dimensional points and corresponding color data.
            valid_mask = position[..., 3] < 1
            points_opengl = position[..., :3][valid_mask]
            points_color = rgba[valid_mask][:, :3]
            # Transform into the world coordinate system.
            points_world = (torch.bmm(
                points_opengl.view(1, -1, 3),
                model_matrix[:3, :3].transpose(0, 1).view(-1, 3, 3),
            ).squeeze(1) + model_matrix[:3, 3])

            # Format color data.
            points_color = torch.clamp(points_color, 0, 1)

            points_world = points_world.squeeze(0)

            # If crop is needed
            if self.pcd_crop:
                min_bound = torch.tensor(self.pcd_crop_bbox[0], dtype=torch.float32).to(device)
                max_bound = torch.tensor(self.pcd_crop_bbox[1], dtype=torch.float32).to(device)
                inside_bounds_mask = (points_world.squeeze(0) >= min_bound).all(dim=1) & (points_world.squeeze(0)
                                                                                          <= max_bound).all(dim=1)
                points_world = points_world[inside_bounds_mask]
                points_color = points_color[inside_bounds_mask]

            # Convert the tensor back to a NumPy array for use with Open3D.
            points_world_np = points_world.cpu().numpy()
            points_color_np = points_color.cpu().numpy()

            if point_num > 0:
                # points_world_np,index = fps(points_world_np,point_num)
                index = index.detach().cpu().numpy()[0]
                points_color_np = points_color_np[index, :]

            return np.hstack((points_world_np, points_color_np))

        if self.head_camera_id is None:
            print("No head camera in static camera list, pointcloud save error!")
            return None

        combined_pcd = np.array([])

        # Merge pointcloud
        if if_combine:
            # combined_pcd = np.vstack((head_pcd , left_pcd , right_pcd, front_pcd))
            if self.collect_wrist_camera:
                combined_pcd = np.vstack((
                    _get_camera_pcd(self.left_camera),
                    _get_camera_pcd(self.right_camera),
                ))
            for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
                if camera_name == "head_camera":
                    if self.collect_head_camera:
                        combined_pcd = np.vstack((combined_pcd, _get_camera_pcd(camera)))
                else:
                    combined_pcd = np.vstack((combined_pcd, _get_camera_pcd(camera)))
        elif self.collect_head_camera:
            combined_pcd = _get_camera_pcd(self.static_camera_list[self.head_camera_id])
        
        def pad_array(numpy_array, target_num):
            current_num = numpy_array.shape[0]
            if current_num < target_num:
                if current_num == 0:
                    numpy_array = np.zeros((target_num, 6))
                else:
                    pad_num = target_num - current_num
                    padding = np.zeros((pad_num, 6))
                    numpy_array = np.vstack([numpy_array, padding])
            return numpy_array

        combined_pcd = pad_array(combined_pcd, self.pcd_down_sample_num)

        pcd_array, index = combined_pcd[:, :3], np.array(range(len(combined_pcd)))

        if self.pcd_down_sample_num > 0:
            pcd_array, index = fps(combined_pcd[:, :3], self.pcd_down_sample_num)
            # fps() 现在已经返回numpy数组，不需要再转换
            if hasattr(index, 'detach'):  # 兼容旧版本可能返回tensor的情况
                index = index.detach().cpu().numpy()[0]
            else:  # 新版本返回numpy数组
                index = index[0] if len(index.shape) > 1 else index

        return combined_pcd[index]

    def get_pcd_from_rgbd(self, if_combine=False, use_fps=False):
        """使用RGB-D重投影生成点云（与训练时完全一致）
        
        🎯 关键：此方法使用与 process_rgb_pcd.py 相同的点云生成逻辑
        确保训练和推理使用完全相同的数据源和处理流程
        
        参数:
            if_combine: 是否合并多个相机的点云
            use_fps: 是否使用FPS采样（False=随机采样，True=FPS采样）
        
        返回:
            np.ndarray: (N, 6) 点云数据 [x, y, z, r, g, b]
        """
        def _get_camera_pcd_rgbd(camera, point_num=0, use_fps_sampling=False):
            # 获取RGB和深度图
            rgba = camera.get_picture_cuda("Color").torch()  # [H, W, 4]
            rgb_img = (rgba[..., :3].cpu().numpy() * 255).astype(np.uint8)  # 转换为0-255范围
            
            # 获取深度图（注意：Sapien的深度图单位是米，需要转换为毫米）
            depth_float = camera.get_picture_cuda("Position").torch()[..., 2]  # Z坐标
            depth_mm = (-depth_float.cpu().numpy() * 1000).astype(np.float32)  # 转换为毫米，注意负号
            
            # 获取相机内参
            intrinsic_matrix = camera.get_intrinsic_matrix()
            
            # 获取相机到世界的变换矩阵
            model_matrix = camera.get_model_matrix()
            
            # 使用RGB-D重投影生成点云
            pointcloud_with_color = depth_to_pointcloud_rgbd(
                depth=depth_mm,
                rgb_img=rgb_img,
                intrinsic=intrinsic_matrix,
                cam2world_gl=model_matrix,
                depth_max=2.0  # 与训练时一致的深度阈值
            )
            
            # 提取XYZ和RGB
            points_world = pointcloud_with_color[:, :3]
            points_color = pointcloud_with_color[:, 3:6]
            
            # 下采样到固定点数
            if point_num > 0 and points_world.shape[0] > 0:
                if use_fps_sampling:
                    # 🔧 使用FPS采样（与训练时process_data_ply.py --use_fps一致）
                    if points_world.shape[0] >= point_num:
                        # 仅使用XYZ进行FPS采样，保持与训练一致
                        _, indices = fps(points_world, num_points=point_num, use_cuda=True)
                        points_world = points_world[indices]
                        points_color = points_color[indices]
                    else:
                        # 上采样：FPS + 随机重复
                        # 仅使用XYZ进行FPS采样
                        _, indices_fps = fps(points_world, num_points=points_world.shape[0], use_cuda=True)
                        
                        indices_random = np.random.choice(points_world.shape[0], point_num - points_world.shape[0], replace=True)
                        
                        points_world_fps = points_world[indices_fps]
                        points_color_fps = points_color[indices_fps]
                        
                        points_world_rnd = points_world[indices_random]
                        points_color_rnd = points_color[indices_random]
                        
                        points_world = np.concatenate([points_world_fps, points_world_rnd], axis=0)
                        points_color = np.concatenate([points_color_fps, points_color_rnd], axis=0)
                else:
                    # 🔧 使用随机采样（与训练时process_data_ply.py默认模式一致）
                    if points_world.shape[0] >= point_num:
                        indices = np.random.choice(points_world.shape[0], point_num, replace=False)
                        points_world = points_world[indices]
                        points_color = points_color[indices]
                    else:
                        # 上采样：随机重复
                        indices = np.random.choice(points_world.shape[0], point_num, replace=True)
                        points_world = points_world[indices]
                        points_color = points_color[indices]
            
            return points_world, points_color

        # 处理单个或多个相机（与原始 get_pcd() 逻辑一致）
        if if_combine:
            # 合并多个相机的点云
            all_points = []
            all_colors = []
            
            if self.collect_wrist_camera:
                left_pts, left_col = _get_camera_pcd_rgbd(self.left_camera, point_num=0, use_fps_sampling=use_fps)
                right_pts, right_col = _get_camera_pcd_rgbd(self.right_camera, point_num=0, use_fps_sampling=use_fps)
                all_points.extend([left_pts, right_pts])
                all_colors.extend([left_col, right_col])
            
            for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
                if camera_name == "head_camera":
                    if self.collect_head_camera:
                        pts, col = _get_camera_pcd_rgbd(camera, point_num=0, use_fps_sampling=use_fps)
                        all_points.append(pts)
                        all_colors.append(col)
                else:
                    pts, col = _get_camera_pcd_rgbd(camera, point_num=0, use_fps_sampling=use_fps)
                    all_points.append(pts)
                    all_colors.append(col)
            
            points_world = np.concatenate(all_points, axis=0)
            points_color = np.concatenate(all_colors, axis=0)
            
            # 合并后进行下采样
            if self.pcd_down_sample_num > 0 and points_world.shape[0] > 0:
                if use_fps:
                    # 使用FPS采样
                    if points_world.shape[0] >= self.pcd_down_sample_num:
                        # 仅使用XYZ进行FPS采样，保持与训练一致
                        _, indices = fps(points_world, num_points=self.pcd_down_sample_num, use_cuda=True)
                        points_world = points_world[indices]
                        points_color = points_color[indices]
                    else:
                        # 上采样
                        # 仅使用XYZ进行FPS采样
                        _, indices_fps = fps(points_world, num_points=points_world.shape[0], use_cuda=True)
                        
                        indices_random = np.random.choice(points_world.shape[0], self.pcd_down_sample_num - points_world.shape[0], replace=True)
                        
                        points_world_fps = points_world[indices_fps]
                        points_color_fps = points_color[indices_fps]
                        
                        points_world_rnd = points_world[indices_random]
                        points_color_rnd = points_color[indices_random]
                        
                        points_world = np.concatenate([points_world_fps, points_world_rnd], axis=0)
                        points_color = np.concatenate([points_color_fps, points_color_rnd], axis=0)
                else:
                    # 使用随机采样
                    if points_world.shape[0] >= self.pcd_down_sample_num:
                        indices = np.random.choice(points_world.shape[0], self.pcd_down_sample_num, replace=False)
                        points_world = points_world[indices]
                        points_color = points_color[indices]
                    else:
                        indices = np.random.choice(points_world.shape[0], self.pcd_down_sample_num, replace=True)
                        points_world = points_world[indices]
                        points_color = points_color[indices]
            
            # 🔧 返回格式：合并XYZ和RGB，返回 [N, 6] 数组（与原始get_pcd()一致）
            combined_pcd = np.concatenate([points_world, points_color], axis=-1)
            return combined_pcd
        else:
            # 只使用head_camera
            if self.head_camera_id is None:
                print("No head camera in static camera list, pointcloud save error!")
                return None
            
            if self.collect_head_camera:
                camera = self.static_camera_list[self.head_camera_id]
                points_world, points_color = _get_camera_pcd_rgbd(camera, point_num=self.pcd_down_sample_num, use_fps_sampling=use_fps)
                
                # 🔧 返回格式：合并XYZ和RGB，返回 [N, 6] 数组（与原始get_pcd()一致）
                combined_pcd = np.concatenate([points_world, points_color], axis=-1)
                return combined_pcd
            else:
                return None
