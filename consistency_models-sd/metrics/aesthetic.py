import argparse
import os

import torch
import pytorch_lightning as pl
import torch.nn as nn
import clip
import pathlib
import numpy as np
import ImageReward as RM
import open_clip
import pandas as pd

from PIL import Image, ImageFile
from tqdm import tqdm

#####  This script will predict the aesthetic score for this image file:

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            # nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


@torch.no_grad()
def calculate_aesthetic_given_paths(paths, max_size):
    _, saving_dir = paths
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    s = torch.load("sac+logos+ava1-l14-linearMSE.pth")
    model.load_state_dict(s)

    model.to("cuda")
    model.eval()

    device = "cuda"
    model2, preprocess = clip.load("ViT-L/14", device=device)  # RN50x64

    path = pathlib.Path(saving_dir)
    files = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in path.glob('*.{}'.format(ext))])[:max_size]

    preds = []
    for file in tqdm(files):
        pil_image = Image.open(file)
        image = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model2.encode_image(image)

        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
        preds.append(prediction.cpu().detach().numpy())

    mean_pred = np.mean(preds)
    return mean_pred


@torch.no_grad()
def calculate_reward_given_paths(path_images, path_prompts):
    model = RM.load("ImageReward-v1.0", device='cuda')
    df = pd.read_csv(path_prompts)
    all_text = list(df['caption'])

    path = pathlib.Path(path_images)
    file_names = sorted([file for ext in IMAGE_EXTENSIONS
                         for file in path.glob('*.{}'.format(ext))])

    named_rewards = {}
    rewards = []
    for file in file_names:
        f = str(file).split('/')[-1]
        idx_text = int(f.split('.')[0])
        prompt = all_text[idx_text]

        file_path = str(file)
        reward = model.score(prompt, [file_path])

        rewards.append(reward)
        named_rewards[file_path] = reward

    print(f'Mean reward {np.mean(rewards)} for {path_images}')

    return named_rewards

@torch.no_grad()
def calculate_clip_given_paths(path_images, path_prompts):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to('cuda')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    df = pd.read_csv(path_prompts)
    all_text = list(df['caption'])

    path = pathlib.Path(path_images)
    file_names = sorted([file for ext in IMAGE_EXTENSIONS
                         for file in path.glob('*.{}'.format(ext))])

    named_rewards = {}
    rewards = []
    for file in file_names:
        f = str(file).split('/')[-1]
        idx_text = int(f.split('.')[0])

        prompt = tokenizer(all_text[idx_text])
        file_path = str(file)

        image = preprocess(Image.open(file_path)).unsqueeze(0).to('cuda')
        image_features = model.encode_image(image)
        text_features = model.encode_text(prompt)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        reward = (image_features.squeeze() * text_features.squeeze()).sum().item()

        rewards.append(reward)
        named_rewards[file_path] = reward

    print(f'Mean reward {np.mean(rewards)} for {path_images}')

    return named_rewards



