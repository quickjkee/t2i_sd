import blobfile as bf
import copy
import functools
import os

import time
from tqdm import tqdm
import torch as th
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam
import shutil

import subprocess
import numpy as np
import copy
import random
import cm.dist_util as dist
import cm.logger as logger
from .fp16_util import MixedPrecisionTrainer
from .nirvana_utils import copy_out_to_snapshot, copy_snapshot_to_out
from evaluations.fid_score import calculate_fid_given_paths
from torchvision.transforms import ToPILImage
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

from .fp16_util import (
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)
        

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        weight_decay=0.0,
        lr_anneal_steps=0
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = RAdam(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist.dev()],
                output_device=dist.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self.step = self.resume_step
        for param_group in self.opt.param_groups:
            param_group['lr'] = self.lr

    def _load_and_sync_parameters(self):
        if dist.get_rank() == 0:
            copy_snapshot_to_out(get_blob_logdir())  # FOR NIRVANA ONLY
        dist.barrier()
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    th.load(resume_checkpoint, map_location=dist.dev()),
                )

        dist.sync_params(self.model.parameters())
        dist.sync_params(self.model.buffers())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, rate)
        print(main_checkpoint, ema_checkpoint)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load(ema_checkpoint, map_location=dist.dev())
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = os.path.join(
            os.path.dirname(main_checkpoint), f"opt.pt"
        )
        if os.path.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            # state_dict = dist.load_state_dict(
            state_dict = th.load(opt_checkpoint, map_location=dist.dev())
            self.opt.load_state_dict(state_dict)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

        
class CMTrainLoop(TrainLoop):
    def __init__(
        self,
        *,
        target_model,
        teacher_model,
        eval_pipe,
        training_mode,
        ema_scale_fn,
        total_training_steps,
        guidance_scale=8.0,
        coco_ref_stats_path=None,
        inception_path=None,
        coco_max_cnt=10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.training_mode = training_mode
        self.ema_scale_fn = ema_scale_fn
        self.target_model = target_model
        self.teacher_model = teacher_model
        self.eval_pipe = eval_pipe
        self.total_training_steps = total_training_steps

        if target_model:
            self.target_model.train()
            self._load_and_sync_target_parameters()
            self.target_model.requires_grad_(False)
        
            self.target_model_params = list(self.target_model.parameters())
            self.target_model_master_params = self.target_model_params

            if self.use_fp16:
                self.target_model_param_groups_and_shapes = get_param_groups_and_shapes(
                    self.target_model.named_parameters()
                )
                self.target_model_master_params = make_master_params(
                    self.target_model_param_groups_and_shapes
                )

        if teacher_model:
            self.teacher_model_params = list(self.teacher_model.parameters())
            self._load_and_sync_teacher_parameters()
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()

        self.timer = 0
        self.global_step = self.step
        self.guidance_scale = guidance_scale
        self.coco_ref_stats_path = coco_ref_stats_path
        self.inception_path = inception_path
        self.coco_max_cnt = coco_max_cnt

    def _load_and_sync_target_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            target_name = name.replace("model", "target_model")
            resume_target_checkpoint = os.path.join(path, target_name)
            if os.path.exists(resume_target_checkpoint) and dist.get_rank() == 0:
                logger.log(
                    f"loading model from checkpoint: {resume_target_checkpoint}..."
                )
                self.target_model.load_state_dict(
                    th.load(resume_target_checkpoint, map_location=dist.dev()),
                )

        dist.sync_params(self.target_model.parameters())
        dist.sync_params(self.target_model.buffers())

    def _load_and_sync_teacher_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            teacher_name = name.replace("model", "teacher_model")
            resume_teacher_checkpoint = os.path.join(path, teacher_name)

            if os.path.exists(resume_teacher_checkpoint) and dist.get_rank() == 0:
                logger.log(
                    f"loading model from checkpoint: {resume_teacher_checkpoint}..."
                )
                self.teacher_model.load_state_dict(
                    th.load(resume_teacher_checkpoint, map_location=dist.dev()),
                )

        dist.sync_params(self.teacher_model.parameters())
        dist.sync_params(self.teacher_model.buffers())

    def run_loop(self):
        saved = False
        while (self.global_step < self.total_training_steps):
            if (self.global_step == 0 or self.global_step % self.save_interval == 0):
                # Sample images and compute fid only on the master node
                inference_steps =[1, 2, 3, 4, 8]
                if self.global_step == 0:
                   inference_steps.append(self.diffusion.num_timesteps)
                for steps in inference_steps:
                    th.cuda.empty_cache()
                    self.generate_coco(num_inference_steps=steps)
                    dist.barrier()

            batch = next(self.data)
            self.run_step(batch['image'], batch['text'])

            saved = False
            if (
                self.global_step
                and self.save_interval != -1
                and self.global_step % self.save_interval == 0
            ):
                
                self.save(save_step=self.global_step % (10 * self.save_interval) == 0)
                saved = True
                th.cuda.empty_cache()

            if self.global_step % self.log_interval == 0:
                logger.dumpkvs()
    
        # Save the last checkpoint if it wasn't already saved.
        if not saved:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            with th.no_grad():
                self._update_ema()
            if self.target_model:
                self._update_target_ema()
            self.step += 1
            self.global_step += 1

        self._anneal_lr()
        self.log_step()

    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with th.no_grad():
            update_ema(
                self.target_model_master_params,
                self.mp_trainer.master_params,
                rate=target_ema,
            )
            if self.use_fp16:
                master_params_to_model_params(
                    self.target_model_param_groups_and_shapes,
                    self.target_model_master_params,
                )

    def forward_backward(self, images, texts):
        self.mp_trainer.zero_grad()
        for i in range(0, images.shape[0], self.microbatch):
            micro_images = images[i : i + self.microbatch].to(dist.dev())
            micro_text = texts[i : i + self.microbatch]
            last_batch = (i + self.microbatch) >= images.shape[0]
            # Diffusion training step
            ema, num_scales = self.ema_scale_fn(self.global_step)
            assert num_scales == 50

            assert self.training_mode == "consistency_distillation"
            compute_losses = functools.partial(
                self.diffusion.consistency_losses,
                self.ddp_model,
                self.teacher_model, 
                self.target_model, 
                image=micro_images,
                prompt=micro_text,
                num_scales=num_scales,
                guidance_scale=self.guidance_scale
            )
            if last_batch or not self.use_ddp:
                losses, t = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses, t = compute_losses()

            loss = losses["loss"].mean()

            log_loss_dict(
                self.diffusion, t, {k: v for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def save(self, save_step=False):
        def save_checkpoint(rate, params, save_step=False):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = "model.pt"
                else:
                    filename = f"ema_{rate}.pt"
                th.save(state_dict, os.path.join(get_blob_logdir(), filename))

                if save_step:
                    if not rate:
                        filename = f"model_{self.global_step}.pt"
                    else:
                        filename = f"ema_{rate}_{self.global_step}.pt"
                    th.save(state_dict, os.path.join(get_blob_logdir(), filename))  

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params, save_step=save_step)

        logger.log("saving optimizer state...")
        if dist.get_rank() == 0:
            th.save(self.opt.state_dict(), os.path.join(get_blob_logdir(), f"opt.pt"))

        if dist.get_rank() == 0:
            if self.target_model:
                logger.log("saving target model state")
                filename = f"target_model.pt"
                th.save(self.target_model.state_dict(), os.path.join(get_blob_logdir(), filename))

            if self.teacher_model:
                logger.log("saving teacher model state")
                filename = f"teacher_model.pt"
                th.save(self.teacher_model.state_dict(), os.path.join(get_blob_logdir(), filename))

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, self.mp_trainer.master_params)
        
        if dist.get_rank() == 0:
            copy_out_to_snapshot(get_blob_logdir())
        dist.barrier()

    def log_step(self):
        step = self.global_step
        logger.logkv("time", time.time() - self.timer)
        logger.logkv("step", step)
        logger.logkv("samples", (step + 1) * self.global_batch)
        self.timer = time.time()

    @th.no_grad()
    def generate_coco(self, scheduler_refining,
                            num_inference_steps=3,
                            num_refining_steps=0,
                            rollback_value=0.3,
                            scheduler_type='DDIM'):

        prev_state_dict = self.model.state_dict()
        self.model.eval()

        timesteps=None

        if num_refining_steps:
            logger.log(f'Refining with rollback value {rollback_value} and steps {num_refining_steps}')

        for ema_rate, params in zip(self.ema_rate, self.ema_params):
            # Setup seed equalt ot the world rank
            generator = th.Generator(device="cuda").manual_seed(dist.get_seed())
            generator_refining = th.Generator(device="cuda").manual_seed(dist.get_seed() + 1) if num_refining_steps else None

            th.manual_seed(dist.get_seed())

            # Load ema params to the model
            ema_state_dict = self.mp_trainer.master_params_to_state_dict(params)
            self.eval_pipe.unet.load_state_dict(ema_state_dict)
            assert not self.eval_pipe.unet.training

            if num_refining_steps or scheduler_type == 'DPM':
                refiner_pipe = copy.deepcopy(self.eval_pipe)
                params = self.teacher_model_params
                refiner_state_dict = self.mp_trainer.master_params_to_state_dict(params)
                refiner_pipe.unet.load_state_dict(refiner_state_dict)
                refiner_pipe.scheduler = scheduler_refining

            if num_inference_steps == 50:
                logger.log(f'Evaluation of the initial diffusion with {num_inference_steps}')
                params = self.teacher_model_params
                teacher_state_dict = self.mp_trainer.master_params_to_state_dict(params)
                self.eval_pipe.unet.load_state_dict(teacher_state_dict)

            dist.barrier()
            
            all_images = []
            all_text_idxs = []
            logger.info(f"Generating coco samples for ema {ema_rate}...")
            rank_batches, rank_batches_index = self.eval_pipe.coco_prompts
            for cnt, mini_batch in enumerate(tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):
                text = list(mini_batch)
                if num_inference_steps in [1, 50]:
                    if scheduler_type == 'DDIM':
                        image = self.diffusion.sample_with_my_step(
                            self.eval_pipe,
                            text,
                            generator=generator,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=self.guidance_scale
                        )
                    elif scheduler_type == 'DPM':
                        image = self.diffusion.sample_with_my_step_dpm(
                            refiner_pipe,
                            text,
                            generator=generator,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=self.guidance_scale
                        )
                else:
                    image, x0_latents, prompt_embeds = self.diffusion.stochastic_iterative_sampler(
                        self.eval_pipe,
                        text, 
                        generator=generator,
                        timesteps=timesteps,
                        num_inference_steps=num_inference_steps, 
                        guidance_scale=self.guidance_scale,
                        with_refining=True if num_refining_steps else False
                    )

                if num_refining_steps:
                    image = self.diffusion.refining(
                        eval_pipe=refiner_pipe,
                        prompt_embeds=prompt_embeds,
                        x0_latents=x0_latents,
                        generator=generator_refining,
                        num_inference_steps=num_refining_steps,
                        guidance_scale=self.guidance_scale,
                        rollback_value=rollback_value,  # [0, 1]
                        scheduler_type=scheduler_type,
                    )

                local_images = []
                local_text_idxs = []
                for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
                    img_tensor = th.tensor(np.array(image[text_idx]))
                    local_images.append(img_tensor)
                    local_text_idxs.append(global_idx)

                local_text_idxs = th.tensor(local_text_idxs).to(dist.dev())
                local_images = th.stack(local_images).to(dist.dev())

                gathered_images = [th.zeros_like(local_images) for _ in range(dist.get_world_size())]
                gathered_text_idxs = [th.zeros_like(local_text_idxs) for _ in range(dist.get_world_size())]

                dist.all_gather(gathered_images, local_images)  # gather not supported with NCCL
                dist.all_gather(gathered_text_idxs, local_text_idxs)  # gather not supported with NCCL

                all_images.extend([sample.cpu().numpy() for sample in gathered_images])
                all_text_idxs.extend([sample.cpu().numpy() for sample in gathered_text_idxs])
              
            if dist.get_rank() == 0:
                save_dir = os.path.join(logger.get_dir(), f"samples_{self.global_step}_steps_{num_inference_steps}_ema_{ema_rate}_ref_{num_refining_steps}")
                os.makedirs(save_dir, exist_ok=True)
                for image, global_idx in zip(all_images, all_text_idxs):
                    ToPILImage()(image).save(os.path.join(save_dir, f"{global_idx}.jpg"))

                subprocess.call(f'CUDA_VISIBLE_DEVICES=0 python3 calc_metrics.py \
                                --folder {save_dir} \
                                --folder_proxy {save_dir} \
                                --folder_csv subset_30k.csv',
                                shell=True)
                shutil.rmtree(f"{save_dir}")

            dist.barrier()

        self.model.load_state_dict(prev_state_dict)
        self.model.train()

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 1  # WAS 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 1  # WAS 0

def recover_resume_step():
    sample_dirs = [name for name in os.listdir(get_blob_logdir()) if "sample" in name]
    last_samples = sorted(sample_dirs, key=lambda x: int(x.split('_')[1]))[-1]
    split = last_samples.split('_')
    if len(split) < 2:
        return 0
    split1 = split[1]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    path = os.path.join(get_blob_logdir(), "model.pt")
    if os.path.exists(path):
        return path
    return None


def find_ema_checkpoint(main_checkpoint, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}.pt"
    path = os.path.join(os.path.dirname(main_checkpoint), filename)
    if os.path.exists(path):
        return path
    return None

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
