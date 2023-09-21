import subprocess
import os
import nirvana_dl
import torch
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
OUTPUT_PATH = get_blob_logdir()

# Try load first
torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

for _ in [6]:
    for _ in [0]: #5, 10, 15, 25, 35, 45
        for _ in [0.5]:
            for (step, ref_step, rollback_v) in [(6, 0, 0.0), (6, 35, 0.6), (6, 35, 0.8), (6, 45, 0.6), (6, 45, 0.8)]:
                print(f'GENERATION WITH CD STEPS {step}, REF STEPS {ref_step}, ROLLBACK V {rollback_v}')
                subprocess.call(f'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.run --standalone \
                                 --nproc_per_node=8 --master-addr=0.0.0.0:1207 scripts/cm_train.py \
                                 --training_mode consistency_distillation \
                                 --target_ema_mode fixed \
                                 --dataset coco \
                                 --start_ema 0.95 \
                                 --scale_mode fixed \
                                 --start_scales 50 \
                                 --total_training_steps 200 \
                                 --loss_norm l2 \
                                 --lr_anneal_steps 0 \
                                 --teacher_model_path {INPUT_PATH}/needed/sd-v1-5 \
                                 --ema_rate 0.9999 \
                                 --global_batch_size 1 \
                                 --lr 0.0008 \
                                 --use_fp16 False \
                                 --weight_decay 0.0 \
                                 --save_interval 10 \
                                 --laion_config configs/laion.yaml \
                                 --weight_schedule uniform \
                                 --coco_train_path coco/coco \
                                 --coco_max_cnt 5000 \
                                 --steps {step} \
                                 --refining_steps {ref_step} \
                                 --resume_checkpoint {INPUT_PATH}/needed/model75000.pt \
                                 --rollback_value {rollback_v} \
                                 --scheduler_type DPM',
                                shell=True)

    #  --resume_checkpoint {INPUT_PATH}/needed/model75000.pt \

                subprocess.call(f'CUDA_VISIBLE_DEVICES=0 python3 calc_metrics.py \
                                --folder tmp/samples_75000_steps_{step}_ema_0.9999_ref_{ref_step}/ \
                                --folder_proxy tmp/samples_75000_steps_{step}_ema_0.9999_ref_0/ \
                                --folder_csv subset_30k.csv',
                                shell=True)

                print('============================================================================================')
                print('============================================================================================')