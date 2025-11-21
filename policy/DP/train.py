"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import hydra, pdb
from omegaconf import OmegaConf
import pathlib, yaml
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    head_camera_type = cfg.head_camera_type
    head_camera_cfg = get_camera_config(head_camera_type)
    # 获取摄像头的高和宽（从配置文件读取）
    cam_h = head_camera_cfg["h"]
    cam_w = head_camera_cfg["w"]
    
    # 设置主摄像头（head_cam）的尺寸
    cfg.task.image_shape = [3, cam_h, cam_w]
    cfg.task.shape_meta.obs.head_cam.shape = [3, cam_h, cam_w]
    print("设置task.image_shape为:", cfg.task.image_shape)
    print("设置head_cam尺寸为:", cfg.task.shape_meta.obs.head_cam.shape)

    # 新增：设置其他摄像头（front/left/right）的尺寸
    # 假设这些摄像头与head_cam型号相同，使用相同的高宽
    # other_cams = ["front_cam", "left_cam", "right_cam"]

    other_cams = ["front_cam"]
    for cam in other_cams:
        if cam in cfg.task.shape_meta.obs:
            cfg.task.shape_meta.obs[cam].shape = [3, cam_h, cam_w]
        else:
            # 若配置中未定义该摄像头，添加默认配置
            cfg.task.shape_meta.obs[cam] = OmegaConf.create({
                "shape": [3, cam_h, cam_w]
            })

    OmegaConf.resolve(cfg)
    cfg.task.image_shape = [3, cam_h, cam_w]
    cfg.task.shape_meta.obs.head_cam.shape = [
        3,
        cam_h,
        cam_w,
    ]

    for cam in other_cams:
        cfg.task.shape_meta.obs[cam].shape = [3, cam_h, cam_w]

    cls = hydra.utils.get_class(cfg._target_)
    print("Training 1:")
    workspace: BaseWorkspace = cls(cfg)
    print(cfg.task.dataset.zarr_path, cfg.task_name)
    workspace.run()


if __name__ == "__main__":
    main()
