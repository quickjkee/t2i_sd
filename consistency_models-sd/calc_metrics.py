import subprocess
import os
import numpy as np
import torch
import argparse
import sys
from metrics.aesthetic import calculate_aesthetic_given_paths
from evaluations.fid_score import calculate_fid_given_paths
from PIL import Image

#SOURCE_CODE_PATH = os.environ['SOURCE_CODE_PATH']
#INPUT_PATH = os.environ['INPUT_PATH']
#OUTPUT_PATH = get_blob_logdir()



parser = argparse.ArgumentParser()
parser.add_argument("--folder", default='', help='number of epochs of training')
conf = parser.parse_args()
folder = conf.folder

#print(calculate_aesthetic_given_paths((0, folder), 50000))

print(calculate_fid_given_paths((folder, 'evaluations/fid_stats_mscoco512_val.npz'),'cuda'))
