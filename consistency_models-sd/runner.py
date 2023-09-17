import subprocess
import os
import nirvana_dl
import torch

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

for step in [6]:
    for ref_step in [15]: #5, 10, 15, 25, 35, 45
        for rollback_v in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
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
                             --resume_checkpoint {INPUT_PATH}/needed/model75000.pt \
                             --steps {step} \
                             --refining_steps {ref_step} \
                             --rollback_value {rollback_v}',
                            shell=True)

            subprocess.call(f'CUDA_VISIBLE_DEVICES=0 python3 calc_metrics.py \
                            --folder tmp/samples_75000_steps_{step}_ema_0.9999/',
                            shell=True)

            subprocess.call(f'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.run --standalone \
                            --nproc_per_node=8 metrics/main_no_faiss.py \
                            --sample_path tmp/samples_75000_steps_{step}_ema_0.9999/ \
                            --real_feature_path {INPUT_PATH}/dinov2_vitl14_laion_1100K_features.pt \
                            --bs 256 \
                            --save_path hz \
                            ', shell=True)

            print('============================================================================================')
            print('============================================================================================')