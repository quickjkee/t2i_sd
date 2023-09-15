export OPENAI_LOGDIR="tmp"
echo $OPENAI_LOGDIR

export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=1

# mpiexec -n 2 python scripts/cm_train.py \
# python -m torch.distributed.run --standalone --nproc_per_node=1 --rdzv_endpoint=0.0.0.0:1206 scripts/cm_train.py \
python3.9 -m torch.distributed.run --standalone --nproc_per_node=1 --master-addr=0.0.0.0:1207 scripts/cm_train.py \
   --training_mode consistency_distillation \
   --target_ema_mode fixed \
   --dataset coco \
   --start_ema 0.95 \
   --scale_mode fixed \
   --start_scales 50 \
   --total_training_steps 200 \
   --loss_norm l2 \
   --lr_anneal_steps 0 \
   --teacher_model_path $INPUT_PATH/needed/pretrained/sd-v1-5 \
   --ema_rate 0.9999 \
   --global_batch_size 1 \
   --lr 0.0008 \
   --use_fp16 False \
   --weight_decay 0.0 \
   --save_interval 10 \
   --laion_config configs/laion.yaml \
   --weight_schedule uniform \
   --coco_train_path coco/coco \
   --coco_max_cnt 16 \
   --resume_checkpoint $INPUT_PATH/needed/model75000.pt \
   --steps 6
   # prev lr 0.000008 # ,0.9999432189950708

CUDA_VISIBLE_DEVICES=0 python3.9 calc_metrics.py \
   --folder tmp/samples_75000_steps_6_ema_0.9999/


