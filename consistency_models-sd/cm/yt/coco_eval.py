import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
from torchvision.datasets import CocoCaptions
from torchvision import transforms


class CocoTextEmbeds(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root

    def __len__(self):
        return 10000

    def __getitem__(self, i):
        data = np.load(os.path.join(self.root, f"{i:05d}.npz"))
        return {"query": data["query"], "text_embedding": data["text_embedding"]}


def cococollate(batch):
    res = [torch.from_numpy(x["text_embedding"]).float() for x in batch]
    queries = [str(x["query"]) for x in batch]
    mask = [torch.ones(x.shape[0], dtype=torch.bool) for x in res]
    return (
        pad_sequence(res, batch_first=True),
        pad_sequence(mask, batch_first=True, padding_value=False),
        queries,
    )


class WrappedCOCO(CocoCaptions):
    def __len__(self):
        return 10000

    def __getitem__(self, index):
        return (super().__getitem__(index)[0]).mul(255).type(torch.uint8)
