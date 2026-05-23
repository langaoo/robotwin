import numpy as np
from dp_model import DP
import yaml
from pathlib import Path
from PIL import Image


def _normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _jet_colormap01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _save_overlay(base_rgb: np.ndarray, heat01: np.ndarray, out_path: Path, alpha: float = 0.45):
    heat_rgb = _jet_colormap01(heat01)
    base = Image.fromarray(base_rgb)
    heat = Image.fromarray(heat_rgb).resize((base.width, base.height), resample=Image.BILINEAR)
    Image.blend(base, heat, alpha=float(alpha)).save(out_path)


def _heat_stats(heat01: np.ndarray) -> dict:
    h, w = heat01.shape
    flat_idx = int(np.argmax(heat01))
    py, px = divmod(flat_idx, w)
    mass = float(np.sum(heat01) + 1e-8)
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    cy = float(np.sum(heat01 * yy) / mass)
    cx = float(np.sum(heat01 * xx) / mass)
    return {
        "peak_x": px,
        "peak_y": py,
        "peak_x_norm": float(px / max(1, w - 1)),
        "peak_y_norm": float(py / max(1, h - 1)),
        "com_x": cx,
        "com_y": cy,
        "com_x_norm": float(cx / max(1, w - 1)),
        "com_y_norm": float(cy / max(1, h - 1)),
    }


def _to_pca1_map(feat_chw: np.ndarray) -> np.ndarray:
    c, h, w = feat_chw.shape
    x = feat_chw.reshape(c, h * w).T  # [HW, C]
    x = x - np.mean(x, axis=0, keepdims=True)
    if c > 1:
        cov = (x.T @ x) / max(1, x.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        p = eigvecs[:, np.argmax(eigvals)]
        y = x @ p
    else:
        y = x[:, 0]
    return y.reshape(h, w)


def _extract_resnet_layer4_feature(resnet, x):
    # x: [1,3,H,W], return: [1,C,h,w]
    x = resnet.conv1(x)
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x)
    x = resnet.layer1(x)
    x = resnet.layer2(x)
    x = resnet.layer3(x)
    x = resnet.layer4(x)
    return x


def _dump_dp_feature_map(model, obs, base_obs, step_idx: int):
    vis_dir = getattr(model, "vis_dump_dir", None)
    if not vis_dir:
        return
    max_calls = int(getattr(model, "vis_dump_max_calls", 0))
    done_calls = int(getattr(model, "_vis_dump_calls", 0))
    if max_calls > 0 and done_calls >= max_calls:
        return

    policy = model.policy
    if not hasattr(policy, "obs_encoder"):
        return
    enc = policy.obs_encoder
    if "head_cam" not in enc.key_model_map or "head_cam" not in enc.key_transform_map:
        return

    resnet = enc.key_model_map["head_cam"]
    transform = enc.key_transform_map["head_cam"]

    x = obs["head_cam"].astype(np.float32)
    x_t = policy.device
    x_tensor = __import__("torch").from_numpy(x).unsqueeze(0).to(device=x_t)
    x_tensor = transform(x_tensor)

    import torch
    with torch.no_grad():
        feat = _extract_resnet_layer4_feature(resnet, x_tensor).squeeze(0).detach().cpu().numpy()  # [C,h,w]

    norm_map = np.linalg.norm(feat, axis=0)
    pca1_map = _to_pca1_map(feat)
    norm01 = _normalize_map(norm_map)
    pca01 = _normalize_map(pca1_map)

    base_rgb = base_obs["observation"]["head_camera"]["rgb"]
    if base_rgb.dtype != np.uint8:
        base_rgb = np.clip(base_rgb * 255.0, 0, 255).astype(np.uint8)

    out_dir = Path(vis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_path = out_dir / f"call{done_calls:04d}_obs0_dp_norm_overlay.jpg"
    pca_path = out_dir / f"call{done_calls:04d}_obs0_dp_pca1_overlay.jpg"
    _save_overlay(base_rgb, norm01, norm_path, alpha=float(getattr(model, "vis_dump_alpha", 0.45)))
    _save_overlay(base_rgb, pca01, pca_path, alpha=float(getattr(model, "vis_dump_alpha", 0.45)))

    stats_path = out_dir / "feature_stats.csv"
    if not stats_path.exists():
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write("call_id,obs_id,model,map_type,peak_x,peak_y,peak_x_norm,peak_y_norm,com_x,com_y,com_x_norm,com_y_norm\n")

    ns = _heat_stats(norm01)
    ps = _heat_stats(pca01)
    with open(stats_path, "a", encoding="utf-8") as f:
        f.write(f"{done_calls},0,dp,norm,{ns['peak_x']},{ns['peak_y']},{ns['peak_x_norm']:.6f},{ns['peak_y_norm']:.6f},{ns['com_x']:.6f},{ns['com_y']:.6f},{ns['com_x_norm']:.6f},{ns['com_y_norm']:.6f}\n")
        f.write(f"{done_calls},0,dp,pca1,{ps['peak_x']},{ps['peak_y']},{ps['peak_x_norm']:.6f},{ps['peak_y_norm']:.6f},{ps['com_x']:.6f},{ps['com_y']:.6f},{ps['com_x_norm']:.6f},{ps['com_y_norm']:.6f}\n")

    model._vis_dump_calls = done_calls + 1

def encode_obs(observation):
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    # left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    # right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    # front_cam = (np.moveaxis(observation["observation"]["front_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        # left_cam=left_cam,
        # right_cam=right_cam,
        # front_cam=front_cam,
    )
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs


def get_model(usr_args):
    ckpt_file = f"./policy/DP/checkpoints/data/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}_h-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
    # ckpt_file = f"./policy/DP/checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"

    action_dim = usr_args['left_arm_dim'] + usr_args['right_arm_dim'] + 2 # 2 gripper
    
    load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)
    
    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']
    
    model = DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)
    model.vis_dump_dir = usr_args.get("vis_dump_dir", None)
    model.vis_dump_max_calls = int(usr_args.get("vis_dump_max_calls", 0))
    model.vis_dump_alpha = float(usr_args.get("vis_dump_alpha", 0.45))
    model._vis_dump_calls = 0
    if model.vis_dump_dir:
        Path(model.vis_dump_dir).mkdir(parents=True, exist_ok=True)
        print(f"[DP] Visual dump enabled: {model.vis_dump_dir}")
    return model


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    _dump_dp_feature_map(model, obs, observation, step_idx=TASK_ENV.take_action_cnt)

    # ======== Get Action ========
    actions = model.get_action(obs)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        _dump_dp_feature_map(model, obs, observation, step_idx=TASK_ENV.take_action_cnt)
        model.update_obs(obs)

def reset_model(model):
    model.reset_obs()
