import subprocess
import os
import nirvana_dl
import torch
import sys
import ImageReward as RM

# Utils
# ----------------------------------------------------------
def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    snapshot_path = nirvana_dl.snapshot.get_snapshot_path()
    nirvana_dl.snapshot.dump_snapshot(snapshot_path)
    return snapshot_path
# ----------------------------------------------------------

SOURCE_CODE_PATH = os.environ['SOURCE_CODE_PATH']
INPUT_PATH = os.environ['INPUT_PATH']
LOG_PATH = os.environ['OPENAI_LOGDIR']
OUTPUT_PATH = get_blob_logdir()

sys.path.append(f'{SOURCE_CODE_PATH}/code/consistency_models-sd')
sys.path.append(f'{SOURCE_CODE_PATH}/code/consistency_models-sd/cm')
sys.path.append(f'{SOURCE_CODE_PATH}/code/consistency_models-sd/metrics')

# Try load first
#torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
#model = RM.load("ImageReward-v1.0", device='cuda')

for _ in [6]:
    for _ in [0]:
        for _ in [0.5]:
            for (step, ref_step, rollback_v) in [(20, 0, 0.0),
                                                 (30, 0, 0.0),
                                                 (40, 0, 0.0),
                                                 (50, 0, 0.0)]:

                print(f'GENERATION WITH CD STEPS {step}, REF STEPS {ref_step}, ROLLBACK V {rollback_v}')
                subprocess.call(f'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.run --standalone \
                                 --nproc_per_node=8 --master-addr=0.0.0.0:1207 scripts/cm_train.py \
                                 --training_mode consistency_distillation \
                                 --target_ema_mode fixed \
                                 --dataset coco \
                                 --start_ema 0.95 \
                                 --scale_mode fixed \
                                 --start_scales 50 \
                                 --total_training_steps 1000000 \
                                 --loss_norm l2 \
                                 --lr_anneal_steps 0 \
                                 --teacher_model_path sd-v1-5 \
                                 --ema_rate 0.9999 \
                                 --global_batch_size 240 \
                                 --microbatch 10 \
                                 --use_fp16 False \
                                 --weight_decay 0.0 \
                                 --save_interval 1000 \
                                 --laion_config configs/laion.yaml \
                                 --weight_schedule uniform \
                                 --coco_max_cnt 5000 \
                                 --steps {step} \
                                 --lr 0.00003 \
                                 --refining_steps {ref_step} \
                                 --coco_path evaluations/coco_subset.csv \
                                 --weight_schedule uniform \
                                 --coco_ref_stats_path evaluations/fid_stats_mscoco256_val.npz \
                                 --inception_path evaluations/pt_inception-2015-12-05-6726825d.pth \
                                 --guidance_scale 8.0 \
                                 --rollback_value {rollback_v} \
                                 --scheduler_type DPM',
                                shell=True)

                for rate in [0.9999]:
                    save_dir = os.path.join(LOG_PATH,
                                            f"samples_{1}_steps_{step}_ema_{rate}_ref_{ref_step}")

                    if rollback_v == 0.0:
                        proxy_dir = save_dir
                    else:
                        proxy_dir = os.path.join(LOG_PATH,
                                                 f"samples_{1}_steps_{step}_ema_{rate}_ref_{0}")

                    subprocess.call(f'CUDA_VISIBLE_DEVICES=0 python3 calc_metrics.py \
                                    --folder {save_dir} \
                                    --folder_proxy {proxy_dir} \
                                    --folder_csv subset_30k.csv',
                                    shell=True)

                print('============================================================================================')
                print('============================================================================================')