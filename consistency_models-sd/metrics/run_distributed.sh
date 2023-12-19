export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=3,7 python -m torch.distributed.run --standalone --nproc_per_node=2 main_no_faiss.py \
    --sample_path /extra_disk_1/quickjkee/results_tmp_delete/samples_75000_steps_5_ema_0.9999/ \
    --real_feature_path /extra_disk_1/dbaranchuk/dpms/dinofeatures/dinov2_vitl14_laion_10M_features.pt \
    --bs 256 \
    --save_path distances_for_check_fake.npz \
