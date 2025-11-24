if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os, sys
import pdb
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib

DP3_ROOT = str(pathlib.Path(__file__).parent.parent)

sys.path.append(DP3_ROOT)
sys.path.append(os.path.join(DP3_ROOT, '3D-Diffusion-Policy'))
sys.path.append(os.path.join(DP3_ROOT, '3D-Diffusion-Policy', 'diffusion_policy_3d'))

from torch.utils.data import DataLoader
import copy

import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
import sys

from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.env_runner.robot_runner import RobotRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler

import pdb, random

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _copy_to_cpu(data):
    """Recursively detach tensors to CPU for checkpoint serialization."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {k: _copy_to_cpu(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        data_type = type(data)
        return data_type(_copy_to_cpu(v) for v in data)
    return data


class TrainDP3Workspace:
    include_keys = ["global_step", "epoch"]
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        # time.sleep(10)
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DP3 = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DP3 = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except:  # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)


       # 配置 optimizer - 根据训练模式设置不同学习率
        if cfg.pretrain.training_mode == "frozen":
            # frozen 模式：冻结backbone参数，不冻结投影层
            optimizer_params = self._get_frozen_optimizer_params()
            self.optimizer = hydra.utils.instantiate(
                cfg.optimizer, 
                params=optimizer_params
            )
            cprint("[TrainDP3] Using FROZEN mode - encoder params excluded from optimizer", "yellow")

        elif cfg.pretrain.training_mode == "finetune":
            optimizer_groups = self._get_finetune_optimizer_groups(cfg)
            self.optimizer = hydra.utils.instantiate(
                cfg.optimizer, 
                params=optimizer_groups,
                _convert_="all"
            )
            cprint("[TrainDP3] Using finetune mode with separate LR for encoder", "green")
        else:  # scratch
            self.optimizer = hydra.utils.instantiate(
                cfg.optimizer, 
                params=self.model.parameters()
            )
            cprint("[TrainDP3] Training from SCRATCH", "magenta")



        # configure training state
        self.global_step = 0
        self.epoch = 0

        # 新增：记录 encoder 是否已解冻（用于 finetune 模式）
        self.encoder_unfrozen = (cfg.pretrain.training_mode != "finetune")
        # 验证冻结状态
        self._verify_freeze_status(cfg)
    
    def _get_frozen_optimizer_params(self):
        """frozen 模式：仅冻结 ULIP backbone，保留 projector 和后续头部可训练"""
        trainable_params = []
        backbone_frozen = 0
        projector_trainable = 0
        other_trainable = 0
        
        for name, param in self.model.named_parameters():
            if 'obs_encoder.extractor.backbone' in name:
                # 确保 backbone 冻结
                param.requires_grad = False
                backbone_frozen += 1
            else:
                if param.requires_grad:
                    trainable_params.append(param)
                    if 'obs_encoder.extractor.projector' in name:
                        projector_trainable += 1
                    else:
                        other_trainable += 1        
        cprint(f"[Frozen] Backbone params frozen: {backbone_frozen}", "yellow")
        cprint(f"[Frozen] Projector params trainable: {projector_trainable}", "cyan")
        cprint(f"[Frozen] Other params trainable: {other_trainable}", "cyan")
        
        return trainable_params

    def _get_finetune_optimizer_groups(self, cfg):
        """为 finetune 模式设置不同的学习率"""
        encoder_lr = cfg.optimizer.lr * cfg.pretrain.finetune_config.encoder_lr_ratio
        
        encoder_backbone_params = []
        other_params = []
        
        # 初始时冻结 encoder
        for name, param in self.model.named_parameters():
            if 'obs_encoder.extractor.backbone' in name:
                param.requires_grad = False  # 先冻结
                encoder_backbone_params.append(param)
            elif param.requires_grad:
                other_params.append(param)
        
        cprint(f"[Finetune] Encoder backbone params: {len(encoder_backbone_params)} (initially frozen)", "cyan")
        cprint(f"[Finetune] Non-backbone params (包含 projector 与动作头): {len(other_params)}", "cyan")
        cprint(f"[Finetune] Encoder LR: {encoder_lr}, Main LR: {cfg.optimizer.lr}", "cyan")
        cprint(f"[Finetune] Will unfreeze at step: {cfg.pretrain.finetune_config.unfreeze_step}", "yellow")
        
        return [
            {'params': encoder_backbone_params, 'lr': encoder_lr, 'name': 'encoder'},
            {'params': other_params, 'lr': cfg.optimizer.lr, 'name': 'head'}
        ]

    def _verify_freeze_status(self, cfg):
        """验证冻结状态是否正确"""
        cprint("\n" + "="*60, "cyan")
        cprint("Verifying Parameter Freeze Status", "cyan")
        cprint("="*60, "cyan")
        
        backbone_trainable = 0
        backbone_frozen = 0
        projector_trainable = 0
        other_trainable = 0
        
        for name, param in self.model.named_parameters():
            if 'obs_encoder.extractor.backbone' in name:
                cprint(f"{name} requires_grad={param.requires_grad}", "yellow")

                if param.requires_grad:
                    backbone_trainable += 1
                else:
                    backbone_frozen += 1
            elif 'obs_encoder.extractor.projector' in name:
                if param.requires_grad:
                    projector_trainable += 1
                else:
                    cprint(f"❌ WARNING: Projector param frozen: {name}", "red")
            elif param.requires_grad:
                other_trainable += 1
        
        cprint(f"Backbone - Trainable: {backbone_trainable}, Frozen: {backbone_frozen}", "yellow")
        cprint(f"Projector - Trainable: {projector_trainable}", "yellow")
        cprint(f"Other - Trainable: {other_trainable}", "cyan")
        
        # 根据模式验证
        if cfg.pretrain.training_mode == "frozen":
            if backbone_trainable > 0:
                cprint(f"❌ WARNING: Frozen mode but {backbone_trainable} encoder params are trainable!", "red")
            else:
                cprint("✓ Frozen mode verified: Backbone encoder params are frozen", "green")
            if projector_trainable == 0:
                cprint("❌ WARNING: Frozen mode should keep projector trainable!", "red")
            else:
                cprint("✓ Frozen mode verified: Projector remains trainable", "green")
                
        elif cfg.pretrain.training_mode == "finetune":
            if backbone_trainable > 0:
                cprint(f"❌ WARNING: Finetune mode but encoder backbone should be initially frozen!", "red")
            else:
                cprint("✓ Finetune mode verified: Encoder backbone initially frozen", "green")
            if projector_trainable == 0:
                cprint("❌ WARNING: Projector is frozen in finetune mode!", "red")
            else:
                cprint("✓ Finetune mode verified: Projector trainable", "green")
                
        elif cfg.pretrain.training_mode == "scratch":
            if backbone_trainable == 0:
                cprint("❌ WARNING: Scratch mode but encoder backbone is frozen!", "red")
            else:
                cprint(f"✓ Scratch mode verified: {backbone_trainable} backbone params trainable", "green")
        
        cprint("="*60 + "\n", "cyan")

    def _maybe_unfreeze_encoder(self, cfg):
        """在 finetune 模式下，达到指定步数后解冻 encoder"""
        if cfg.pretrain.training_mode != "finetune":
            return
        
        if self.encoder_unfrozen:
            return
        
        unfreeze_step = cfg.pretrain.finetune_config.unfreeze_step
        
        if self.global_step >= unfreeze_step:
            cprint(f"[Finetune] Unfreezing encoder at step {self.global_step}", "green")
            for name, param in self.model.named_parameters():
                if 'obs_encoder.extractor.backbone' in name:
                    param.requires_grad = True
            
            extractor = getattr(self.model.obs_encoder, 'extractor', None)
            if hasattr(extractor, 'set_backbone_train_mode'):
                extractor.set_backbone_train_mode(True)
                cprint("[Finetune] Encoder backbone switched to train() for BN stats", "green")

            self.encoder_unfrozen = True
            cprint("[Finetune] Encoder unfrozen, now trainable", "green")
            # 验证解冻成功
            self._verify_freeze_status(cfg)

    def _get_checkpoint_dir_name(self, cfg):
        """生成 checkpoint 目录名称，包含训练模式信息"""
        base_name = f"{cfg.task.name}_{cfg.training.seed}"
        
        # 添加点云类型标识
        if cfg.policy.use_pc_color:
            base_name += "_rgb"
        
        # 添加 pointnet 类型标识
        pointnet_type = cfg.policy.pointnet_type
        if pointnet_type == "ulip_rgbxyz":
            base_name += "_ulip_rgbxyz"
        elif pointnet_type == "ulip_xyz":
            base_name += "_ulip_xyz"
        elif pointnet_type == "pointnet":
            base_name += "_pointnet"
        
        # 添加训练模式标识
        mode = cfg.pretrain.training_mode
        if mode == "frozen":
            base_name += "_frozen"
        elif mode == "finetune":
            lr_ratio = cfg.pretrain.finetune_config.encoder_lr_ratio
            base_name += f"_ft{lr_ratio}"
        elif mode == "scratch":
            base_name += "_scratch"
        
        return base_name

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        # 打印训练模式信息
        self._print_training_mode(cfg)

        WANDB = False

        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False

        RUN_ROLLOUT = False
        RUN_VALIDATION = True  # reduce time cost

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            print(f"Looking for checkpoint at: {lastest_ckpt_path}")
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

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

        env_runner = None

        cfg.logging.name = str(cfg.task.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        if WANDB:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging,
            )
            wandb.config.update({
                "output_dir": self.output_dir,
            })

        # configure checkpoint
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
        checkpoint_num = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        # 修复：确保断点续训时不会重复训练已完成的epoch
        # 如果resume，从self.epoch开始；否则从0开始
        start_epoch = self.epoch
        for local_epoch_idx in range(start_epoch, cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()

                   # 检查是否需要解冻 encoder (仅 finetune 模式)
                    if cfg.pretrain.training_mode == "finetune":
                        self._maybe_unfreeze_encoder(cfg)


                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    t1_1 = time.time()
                    raw_loss, loss_dict = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    if cfg.pretrain.training_mode == "frozen" and batch_idx == 0 and self.epoch == 0:
                        self._verify_frozen_gradients()

                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        # 可选：为 encoder 设置梯度裁剪
                        if cfg.pretrain.training_mode == "finetune" and self.encoder_unfrozen:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.obs_encoder.extractor.parameters(), 
                                max_norm=1.0
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        "train_loss": raw_loss_cpu,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }

                    # 如果使用微调，记录 encoder 的学习率和状态
                    if cfg.pretrain.training_mode == "finetune":
                        if len(self.optimizer.param_groups) > 1:
                            step_log["encoder_lr"] = self.optimizer.param_groups[0]['lr']
                            step_log["head_lr"] = self.optimizer.param_groups[1]['lr']
                        step_log["encoder_frozen"] = not self.encoder_unfrozen
                    
                    
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()

                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = batch_idx == (len(train_dataloader) - 1)
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        if WANDB:
                            wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
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

            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, loss_dict = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            print(f"epoch {self.epoch}, eval loss: ", float(loss.cpu()))
                            if (cfg.training.max_val_steps
                                    is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log["val_loss"] = val_loss

            # # checkpoint
            # if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:

            #     if not cfg.policy.use_pc_color:
            #         if not os.path.exists(f"checkpoints/{self.cfg.task.name}_{cfg.training.seed}"):
            #             os.makedirs(f"checkpoints/{self.cfg.task.name}_{cfg.training.seed}")
            #         save_path = f"checkpoints/{self.cfg.task.name}_{cfg.training.seed}/{self.epoch + 1}.ckpt"
            #     else:
            #         if not os.path.exists(f"checkpoints/{self.cfg.task.name}_w_rgb_{cfg.training.seed}"):
            #             os.makedirs(f"checkpoints/{self.cfg.task.name}_w_rgb_{cfg.training.seed}")
            #         save_path = f"checkpoints/{self.cfg.task.name}_w_rgb_{cfg.training.seed}/{self.epoch + 1}.ckpt"

            #     self.save_checkpoint(save_path)

            # checkpoint - 使用新的命名策略
            if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                checkpoint_dir = f"checkpoints/{self._get_checkpoint_dir_name(cfg)}"
                
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                
                save_path = f"{checkpoint_dir}/{self.epoch + 1}.ckpt"
                self.save_checkpoint(save_path)

            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            if WANDB:
                wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

    def _verify_frozen_gradients(self):
        """验证 frozen encoder backbone 的梯度确实为 None/0"""
        encoder_has_grad = False
        for name, param in self.model.named_parameters():
            if 'obs_encoder.extractor.backbone' in name:
                if param.grad is not None and param.grad.abs().sum() > 0:
                    cprint(f"⚠️  WARNING: {name} has non-zero gradient!", "red")
                    encoder_has_grad = True
        
        if not encoder_has_grad:
            cprint("✓ Gradient check passed: Frozen encoder has no gradients", "green")
        else:
            cprint("❌ Gradient check failed: Some encoder params have gradients!", "red")

    def _print_training_mode(self, cfg):
        """打印当前训练模式信息"""
        cprint("=" * 60, "cyan")
        cprint("Training Configuration", "cyan")
        cprint("=" * 60, "cyan")
        cprint(f"Point Cloud Type: {cfg.policy.pointnet_type}", "yellow")
        cprint(f"Use RGB: {cfg.policy.use_pc_color}", "yellow")
        cprint(f"Training Mode: {cfg.pretrain.training_mode}", "green")
        
        if cfg.pretrain.training_mode == "finetune":
            cprint(f"  Encoder LR Ratio: {cfg.pretrain.finetune_config.encoder_lr_ratio}", "cyan")
            cprint(f"  Warmup Steps: {cfg.pretrain.finetune_config.warmup_steps}", "cyan")
        
        cprint("=" * 60, "cyan")

    def get_policy_and_runner(self, cfg, usr_args):
        # load the latest checkpoint

        cfg = copy.deepcopy(self.cfg)

        n_obs_steps = cfg['n_obs_steps']
        n_action_steps = cfg['n_action_steps']

        env_runner = RobotRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)

        if not cfg.policy.use_pc_color:
            ckpt_file = pathlib.Path(
                os.path.join(
                    DP3_ROOT,
                    # f"./checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}_{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
                    f"./checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}_{usr_args['seed']}_{cfg.policy.pointnet_type}_scratch/{usr_args['checkpoint_num']}.ckpt"

                ))
        else:
            ckpt_file = pathlib.Path(
                os.path.join(
                    DP3_ROOT,
                    # f"./checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}_w_rgb_{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
                    f"./checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}_{usr_args['seed']}_rgb_{cfg.policy.pointnet_type}_frozen/{usr_args['checkpoint_num']}.ckpt"

                ))
        assert ckpt_file.is_file(), f"ckpt file doesn't exist, {ckpt_file}"

        if ckpt_file.is_file():
            cprint(f"Resuming from checkpoint {ckpt_file}", "magenta")
            self.load_checkpoint(path=ckpt_file)

        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()
        return policy, env_runner

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        use_thread=False,
    ):
        print("saved in ", path)
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir", )

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {"cfg": self.cfg, "state_dicts": dict(), "pickles": dict()}

        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload["state_dicts"][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload["state_dicts"][key] = value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)

        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest"):
        if tag == "latest":
            return pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        elif tag == "best":
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath("checkpoints")
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if "latest" in ckpt:
                    continue
                score = float(ckpt.split("test_mean_score=")[1].split(".ckpt")[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath("checkpoints", best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(self, path=None, tag="latest", exclude_keys=None, include_keys=None, **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(cls, path, exclude_keys=None, include_keys=None, **kwargs):
        payload = torch.load(open(path, "rb"), pickle_module=dill)
        instance = cls(payload["cfg"])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs,
        )
        return instance

    def save_snapshot(self, tag="latest"):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath("snapshots", f"{tag}.pkl")
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, "rb"), pickle_module=dill)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy_3d", "config")),
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
