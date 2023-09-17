import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
from src.data import get_dinov2_loader
from src import dist_util
from src.nirvana_utils import copy_out_to_snapshot
import torch.distributed as dist

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
sys.path.append(f'{SOURCE_CODE_PATH}/t2i_sd/consistency_models-sd/metrics')
######################################################


parser = argparse.ArgumentParser(description='Compute kNN distance to DINOv2 Embeddings on LAION')
parser.add_argument('--bs', type=int, default=500)
parser.add_argument('--save_path', type=str, default='./distances.npz')
parser.add_argument('--sample_path', type=str)
parser.add_argument('--real_feature_path', type=str)


@torch.no_grad()
def extract_features(loader, model):
    """ Extract DINOv2 features from generated samples """
    d = model.norm.weight.shape[0]
    features = torch.zeros(len(loader.dataset), d, dtype=torch.float16, device='cuda')
    counter = 0
    for batch in tqdm(loader):
        with torch.cuda.amp.autocast():
            out = model(batch.cuda())
        out /= out.norm(dim=-1, keepdim=True)
        features[counter: counter + loader.batch_size] = out.half()
        counter += loader.batch_size
    return features
    

def main():
    args = parser.parse_args()
    torch.set_num_threads(40)
    dist_util.init()

    # Extract DINOv2 features from generated samples
    if dist.get_rank() == 0:
        print('Loading DINOv2 ViT-L/14 model...')
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dinov2_vitl14 = dinov2_vitl14.eval().to(dist_util.dev())

    loader = get_dinov2_loader(args.sample_path, batch_size=args.bs)
    if dist.get_rank() == 0:
        print(f'Extracting features from {args.sample_path}')
    fake_features = extract_features(loader, dinov2_vitl14)

    assert os.path.exists(args.real_feature_path)
    real_features = torch.load(args.real_feature_path)
    shard_size = len(real_features) // dist.get_world_size()
    shard_real_features = real_features.split(shard_size)[dist.get_rank()].to(dist_util.dev())

    # Compute distances
    if dist.get_rank() == 0:
        print('Computing distances...')
    max_distances = (fake_features @ shard_real_features.T).max(-1, keepdim=True)[0]
    assert max_distances.shape[0] == len(fake_features) and max_distances.shape[1] == 1
    dist.barrier()

    gathered_max_distances = [torch.zeros_like(max_distances) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_max_distances, max_distances)  
    all_max_distances = torch.cat(gathered_max_distances, dim=1)
    final_max_distances = all_max_distances.max(-1)[0]

    # Save results
    if dist.get_rank() == 0:
        print(f"Saving results to {args.save_path}...")
    if dist.get_rank() == 0:
        os.makedirs(args.save_path, exist_ok=True)
        torch.save(
            {
                "image_path": loader.dataset.img_files,
                "distances": final_max_distances,
            },  
            os.path.join(args.save_path, 'dinov2_distances.pt')
        )
        copy_out_to_snapshot(args.save_path)

    if dist.get_rank() == 0:
        print(torch.mean(final_max_distances))
    

if __name__ == '__main__':
    main()
