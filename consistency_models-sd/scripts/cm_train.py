"""
Train a diffusion model on images.
"""
import sys
import os

# Necessary stuff
######################################################
SOURCE_CODE_PATH = os.environ['SOURCE_CODE_PATH']
INPUT_PATH = os.environ['INPUT_PATH']

sys.path.append(f'{SOURCE_CODE_PATH}/t2i_sd')
sys.path.append(f'{SOURCE_CODE_PATH}/t2i_sd/consistency_models-sd')
sys.path.append(f'{SOURCE_CODE_PATH}/t2i_sd/consistency_models-sd/cm')
sys.path.append(f'{SOURCE_CODE_PATH}/t2i_sd/consistency_models-sd/scripts')
######################################################

import argparse
import torch as th
import yaml
import os
from omegaconf import OmegaConf
from cm.yt.utils import instantiate_from_config

from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from cm import logger
from cm.script_util import (
    set_dropout_rate,
    cm_train_defaults,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util import CMTrainLoop
import cm.dist_util as dist
from cm.diffusion import DenoiserSD
from copy import deepcopy

from coco_dataset import COCODataset, InfiniteSampler


def prepare_coco_prompts(path, bs=20, max_cnt=10000):
    import pandas as pd
    import numpy as np

    df = pd.read_csv(path)
    all_text = list(df['caption'])
    all_text = all_text[:max_cnt]

    num_batches = ((len(all_text) - 1) // (bs * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = all_batches[dist.get_rank():: dist.get_world_size()] 

    index_list = np.arange(len(all_text))
    print(len(index_list))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]
    return rank_batches, rank_batches_index


def main():
    args = create_argparser().parse_args()

    dist.init()
    logger.configure()
    th.set_num_threads(40)

    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )

    # Create main pipe
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.teacher_model_path, 
        torch_dtype=th.float16 if args.use_fp16 else th.float32, 
        variant="fp16" if args.use_fp16 else "fp32", 
    )
    if args.scheduler_type == 'DDIM':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler_type == 'DPM':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.scheduler.final_alpha_cumprod = th.tensor(1.0) # set boundary condition
    # ^-- Keep this in mind, may be important
    pipe.to(dist.dev())
    model = pipe.unet.train()

    # Create eval pipe that uses distilled EMA for sampling 
    eval_pipe = StableDiffusionPipeline(
        pipe.vae,
        pipe.text_encoder,
        pipe.tokenizer,
        pipe.unet,
        pipe.scheduler,
        None, # safety_checker
        None, # feature_extractor
        requires_safety_checker = False
    )

    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size
        
    #############
    # Load data #
    #############

    # Train dataset
    #with open(args.laion_config, "r") as f:
    #    config = yaml.load(f, Loader=yaml.SafeLoader)
    #    config = OmegaConf.create(config)
    #    config['train_dataloader'][0]['params']['batch_size'] = batch_size

    if args.dataset == 'laion':
        data = instantiate_from_config(config['train_dataloader'][0])
    elif args.dataset == 'coco':
        data = None
        #from torchvision import transforms
        #transform = transforms.Compose([
        #    transforms.Resize(512),
        #    transforms.RandomCrop(512),
        #    transforms.ToTensor(),
        #    lambda x: 2 * x - 1
        #])
        #dataset = COCODataset(args.coco_train_path, subset_name='train2014', transform=transform)
        #dataset_sampler = InfiniteSampler(dataset=dataset, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=0)
        #data = iter(th.utils.data.DataLoader(
        #    dataset=dataset, sampler=dataset_sampler, batch_size=batch_size)
        #)
    else:
        raise(f"Unsupported dataset {args.dataset}...")
    
    eval_pipe.coco_prompts = prepare_coco_prompts(args.coco_path, max_cnt=args.coco_max_cnt)

    # load the target and teacher models for distillation, if path specified.
    logger.log(f"loading the teacher model from {args.teacher_model_path}")
    # teacher_model = deepcopy(model).to(dist.dev())
    teacher_model = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.teacher_model_path, 
        torch_dtype=th.float16, 
        variant="fp16", 
    ).unet.to(dist.dev())

    logger.log("creating the target model")
    target_model = deepcopy(model).to(dist.dev())

    # Check that all models have the same parameters
    for dst, src in zip(target_model.parameters(), model.parameters()):
        assert (dst.data == src.data).all()

    # for dst, src in zip(teacher_model.parameters(), model.parameters()):
    #     assert (dst.data == src.data).all()
    
    assert len(list(target_model.buffers())) == len(list(model.buffers())) == len(list(teacher_model.buffers())) == 0 

    # Create SD denoiser
    diffusion = DenoiserSD(
        pipe,
        sigma_data = 0.5,
        loss_norm = args.loss_norm,
        num_timesteps=args.start_scales,
        weight_schedule=args.weight_schedule,
        use_fp16=args.use_fp16
    )

    CMTrainLoop(
        model=model,
        diffusion=diffusion,
        target_model=target_model,
        teacher_model=teacher_model,
        eval_pipe=eval_pipe,
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
        data=data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        guidance_scale=args.guidance_scale,
        # Eval fid
        coco_ref_stats_path=args.coco_ref_stats_path,
        inception_path=args.inception_path,
        coco_max_cnt=args.coco_max_cnt
    ).generate_coco(args.steps,
                    num_refining_steps=args.refining_steps,
                    rollback_value=args.rollback_value,
                    scheduler=args.scheduler_type)

def create_argparser():
    defaults = dict(
        data_dir="",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=20,
        save_interval=2000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='laion', 
        laion_config="configs/laion.yaml",
        coco_train_path=".",
        coco_path="subset_30k.csv",
        weight_schedule='uniform',
        teacher_dropout=0.0,
        guidance_scale=8.0,
        coco_max_cnt=10000,
        steps=6,
        refining_steps=5,
        rollback_value=0.3,
        scheduler_type='DDIM',
        # Eval fid
        coco_ref_stats_path="evaluations/fid_stats_mscoco256_val.npz",
        inception_path="evaluations/pt_inception-2015-12-05-6726825d.pth"
    )
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
