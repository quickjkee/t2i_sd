import subprocess
import os
import numpy as np
import torch
import argparse
import sys
from metrics.aesthetic import calculate_aesthetic_given_paths, calculate_reward_given_paths, calculate_clip_given_paths
from evaluations.fid_score import calculate_fid_given_paths
from PIL import Image
import shutil

SOURCE_CODE_PATH = os.environ['SOURCE_CODE_PATH']
INPUT_PATH = os.environ['INPUT_PATH']


# Move images w.r.t metric
# --------------------------------------
def mover(folder_proxy, reward_proxy, folder, percentile):
    # percentile - [0, 1]

    # Configuration
    outdir = f'refining_{percentile}'
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    total_size = len(reward_proxy)
    os.makedirs(outdir)

    print(f'Refining part - {percentile * 100}%,  Untouched part - {100 - percentile * 100}% \n'
          f'Refining folder - {folder}, Proxy folder - {folder_proxy} \n'
          f'Saving to {outdir}')

    metric_dict = reward_proxy
    sorted_metric_dict = dict(sorted(metric_dict.items(), key=lambda item: item[1], reverse=False))

    names = list(sorted_metric_dict.keys())
    ref_names = names[:int(total_size * percentile)]
    left_names = [name for name in names if name not in ref_names]

    root_refining = folder
    for file in ref_names:
        name_splitted = file.split('/')[-1]
        shutil.copy(f'{root_refining}/{name_splitted}', f'{outdir}/{name_splitted}')

    for file in left_names:
        name_splitted = file.split('/')[-1]
        shutil.copy(f'{file}', f'{outdir}/{name_splitted}')

    print(f'Size of adaptive folder {len(os.listdir(outdir))}')
    return outdir
# --------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--folder", default='', help='number of epochs of training')
parser.add_argument("--folder_csv", default='', help='number of epochs of training')
parser.add_argument("--folder_proxy", default='', help='number of epochs of training')
conf = parser.parse_args()
folder = conf.folder
folder_csv = conf.folder_csv
folder_proxy = conf.folder_proxy


#torch.cuda.empty_cache()

# Not an adaptive case
if folder == folder_proxy:
    reward = calculate_clip_given_paths(folder, folder_csv)

    #fid = calculate_fid_given_paths((folder, 'evaluations/fid_stats_mscoco512_val.npz'), 'cuda')
    #print(f'Fid {fid}')

else:
    print(f'Adaptive metric, Proxy {folder_proxy}, original {folder}')
    reward_proxy = calculate_clip_given_paths(folder_proxy, folder_csv)

    for perc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        folder_adaptive = mover(folder_proxy, reward_proxy, folder, percentile=perc)
        reward = calculate_clip_given_paths(f'{folder_adaptive}/', folder_csv)
