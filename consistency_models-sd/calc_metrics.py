import subprocess
import os
import numpy as np
import torch
import argparse
import sys
from metrics.aesthetic import calculate_aesthetic_given_paths, calculate_reward_given_paths
from evaluations.fid_score import calculate_fid_given_paths
from PIL import Image

SOURCE_CODE_PATH = os.environ['SOURCE_CODE_PATH']
INPUT_PATH = os.environ['INPUT_PATH']


parser = argparse.ArgumentParser()
parser.add_argument("--folder", default='', help='number of epochs of training')
parser.add_argument("--folder_csv", default='', help='number of epochs of training')
conf = parser.parse_args()
folder = conf.folder
folder_csv = conf.folder_csv

#aesth = calculate_aesthetic_given_paths((0, folder), 50000)
#print(f'Aesthetic {aesth}')

#fid = calculate_fid_given_paths((folder, 'evaluations/fid_stats_mscoco512_val.npz'), 'cuda')
#print(f'Fid {fid}')

reward = calculate_reward_given_paths(folder, folder_csv)
print(f'Reward {reward}')