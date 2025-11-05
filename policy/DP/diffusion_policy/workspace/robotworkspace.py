if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy

import tqdm, random
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class RobotWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        seed = cfg.training.seed
        head_camera_type = cfg.head_camera_type

        # ========== one-time display of used camera keys (from shape_meta) ==========
        try:
            obs_meta = cfg.shape_meta.get("obs", {})
            used_rgb_cams = [k for k, v in obs_meta.items() if v.get("type", "low_dim") == "rgb"]
            if len(used_rgb_cams) > 0:
                print(f"[Training] 使用的摄像头: {', '.join(used_rgb_cams)}")
            else:
                print("[Training] 未在 shape_meta 中发现 rgb 相机键")
        except Exception as e:
            print(f"[Training] 无法解析 shape_meta 中的摄像头信息: {e}")

        # --------------------------
        # 1. 微调模式（优先级高于断点续训）
        # --------------------------
        if cfg.finetune.isfinetune and cfg.finetune.resume_from:
            pretrained_path = pathlib.Path(cfg.finetune.resume_from)
            print()
            if not pretrained_path.exists():
                raise FileNotFoundError(f"预训练模型不存在: {pretrained_path}")
            
            try:
                # 加载预训练模型权重（仅加载模型，不加载优化器/训练状态）
                ckpt = torch.load(pretrained_path, map_location="cpu")
                # 尝试从检查点中提取模型参数
                # 从检查点中提取模型参数（处理两种常见的检查点格式）
                if "state_dicts" in ckpt:
                    # 处理工作区格式的检查点（包含多个组件的状态）
                    state = ckpt["state_dicts"].get("model", ckpt["state_dicts"])
                elif "model" in ckpt:
                    state = ckpt["model"]
                else:
                    state = ckpt  # 假设整个文件都是模型参数
                self.model.load_state_dict(state)
      
                print(f"微调模式：加载预训练模型 {pretrained_path}")
                print(f"微调模式：基础模型{cfg.finetune.base_model}")

                # 加载 EMA 模型（如果使用）
                if self.ema_model is not None:
                    self.ema_model.load_state_dict(state)

                # 微调时冻结指定层（如编码器）
                if cfg.finetune.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)
                    print("微调模式：冻结编码器权重")

                # 微调时重置优化器（避免沿用预训练的优化器状态）
                if cfg.finetune.reset_optimizer:
                    self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
                    print("微调模式：重置优化器")

                # 微调时重置训练状态（epoch/global_step 从零开始）
                self.epoch = 0
                self.global_step = 0
                print("微调模式：重置训练状态（epoch/global_step = 0）")

            except Exception as e:
                print(f"微调模型加载失败: {e}")
                raise

        # --------------------------
        # 2. 断点续训模式（仅当未启用微调时生效）
        # --------------------------
        else:
            resume_path = None
            # 优先使用显式指定的续训路径
            if getattr(cfg.training, "resume_from", None):
                resume_path = pathlib.Path(cfg.training.resume_from)
            # 否则使用默认的最新 checkpoint
            elif cfg.training.resume:
                latest_ckpt_path = self.get_checkpoint_path()
                if latest_ckpt_path.is_file():
                    resume_path = latest_ckpt_path

            # 处理目录路径（自动找最新 ckpt）
            if resume_path and resume_path.is_dir():
                candidates = sorted(
                    [p for p in resume_path.glob("*") if p.suffix in [".ckpt", ".pth", ".pt"]],
                    key=lambda p: p.stat().st_mtime
                )
                resume_path = candidates[-1] if candidates else None

            if resume_path and resume_path.is_file():
                print(f"断点续训：从 {resume_path} 加载")
                try:
                    # 加载完整训练状态（模型、优化器、epoch/global_step）
                    self.load_checkpoint(path=str(resume_path))
                    # 从文件名推断 epoch（如果 checkpoint 中未存储）
                    if self.epoch == 0:
                        try:
                            self.epoch = int(resume_path.stem)
                        except:
                            pass
                    print(f"续训状态：epoch={self.epoch}, global_step={self.global_step}")
                except Exception as e:
                    print(f"续训加载失败: {e}")
                    raise


        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = create_dataloader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) //
            cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        env_runner = None

        # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        print("日志输出路径："+str(self.output_dir))
        save_name = cfg.training.save_path if cfg.training.save_path  else pathlib.Path(self.cfg.task.dataset.zarr_path).stem
        print(f"保存路径: checkpoints/{save_name}-{seed}")
        
        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"),
                                             **cfg.checkpoint.topk)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        # helper: compute global grad norm (L2)
        def _compute_grad_norm(model):
            total = 0.0
            has_grad = False
            for p in model.parameters():
                if p.grad is not None:
                    has_grad = True
                    # use .detach() to avoid graph
                    g = p.grad.detach()
                    total += float(torch.sum(g * g).item())
            if not has_grad:
                return 0.0
            return float(total) ** 0.5

        prev_grad_norm = None

        with JsonLogger(log_path) as json_logger:
            while self.epoch < cfg.training.num_epochs:
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # compute gradient norm after backward for logging
                        try:
                            grad_norm = _compute_grad_norm(self.model)
                        except Exception:
                            grad_norm = None
                        grad_delta = None
                        if prev_grad_norm is not None and grad_norm is not None:
                            grad_delta = grad_norm - prev_grad_norm

                        # step optimizer
                        if (self.global_step % cfg.training.gradient_accumulate_every == 0):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        postfix = {"loss": round(raw_loss_cpu, 6)}
                        if grad_norm is not None:
                            postfix["grad"] = round(float(grad_norm), 6)
                        if grad_delta is not None:
                            postfix["dgrad"] = round(float(grad_delta), 6)
                        tepoch.set_postfix(postfix, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        # attach gradient stats to log
                        if grad_norm is not None:
                            step_log["grad_norm_l2"] = grad_norm
                        if grad_delta is not None:
                            step_log["grad_norm_delta"] = grad_delta
                        prev_grad_norm = grad_norm if grad_norm is not None else prev_grad_norm

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps
                                is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                                val_dataloader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps
                                        is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    # save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                    save_name = cfg.training.save_path if cfg.training.save_path  else pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                    if cfg.finetune.isfinetune:
                        self.save_checkpoint(f"checkpoints/{save_name}-{cfg.finetune.base_model}-{seed}/{self.epoch + 1}.ckpt")  # TODO

                    else:
                        self.save_checkpoint(f"checkpoints/{save_name}-{seed}/{self.epoch + 1}.ckpt")  # TODO

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


class BatchSampler:

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
):
    batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)

    def collate(x):
        assert len(x) == 1
        return x[0]

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    return dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
