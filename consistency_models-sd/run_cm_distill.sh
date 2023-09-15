export OPENAI_LOGDIR="${PWD}/../tmp/ddim_bs80"
echo $OPENAI_LOGDIR

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

# mpiexec -n 2 python scripts/cm_train.py \
python -m torch.distributed.run --standalone --nproc_per_node=1 --master-addr=0.0.0.0:1208 scripts/cm_train.py \
   --training_mode consistency_distillation \
   --target_ema_mode fixed \
   --start_ema 0.95 \
   --scale_mode fixed \
   --start_scales 50 \
   --total_training_steps 30000 \
   --loss_norm l2 \
   --lr_anneal_steps 0 \
   --teacher_model_path pretrained/sd-v1-5 \
   --ema_rate 0.999,0.9999,0.9999432189950708 \
   --global_batch_size 1 \
   --lr 0.00003 \
   --use_fp16 False \
   --weight_decay 0.0 \
   --save_interval 100 \
   --laion_config configs/laion.yaml \
   --weight_schedule uniform \
   --guidance_scale 8.0 \
   --coco_max_cnt 16