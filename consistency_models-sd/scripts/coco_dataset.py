import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import csv


def imagenet_transform(size):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop([size, size]),
        transforms.ToTensor(),
        normalize,])


def adaptive_image_resize(x, h, w):
    if x.size[0] > x.size[1]:
        t = transforms.Resize(h)
    else:
        t = transforms.Resize(w)
    return t(x)


class COCODataset(Dataset):
    def __init__(self, root_dir, subset_name='subset', transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
        sample_dir = os.path.join(root_dir, subset_name)
        self.samples = sorted([os.path.join(sample_dir, fname) for fname in os.listdir(sample_dir) 
                        if fname[-4:] in self.extensions], key=lambda x: x.split('/')[-1].split('.')[0])
        
        self.captions = {}
        with open(os.path.join(root_dir, f"{subset_name}.csv"), newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(spamreader):
                if i == 0:
                    continue
                self.captions[row[1]] = row[2]
        for i in range(10):
            print(self.samples[i], self.captions[os.path.basename(self.samples[i])])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_path = self.samples[idx]
        sample = Image.open(sample_path).convert('RGB')

        if self.transform:
            sample = self.transform(sample)

        return {'image': sample, 'text': self.captions[os.path.basename(sample_path)]}
    

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1
    

class LabeledDatasetImagesExtractor(Dataset):
    def __init__(self, ds, img_field=0):
        self.source = ds
        self.img_field = img_field

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        return self.source[item][self.img_field]


class DatasetLabelWrapper(Dataset):
    def __init__(self, ds, label, transform=None):
        self.source = ds
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        img = self.source[item]
        if self.transform is not None:
            img = self.transform(img)
        return (img, self.label[item])


class TransformedDataset(Dataset):
    def __init__(self, source, transform, img_index=0):
        self.source = source
        self.transform = transform
        self.img_index = img_index

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        out = self.source[index]
        if isinstance(out, tuple):
            return self.transform(out[self.img_index]), out[1 - self.img_index]
        else:
            return self.transform(out)


class TensorsDataset(Dataset):
    def __init__(self, source_dir):
        self.source_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir)\
            if f.endswith('.pt')]

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, index):
        return torch.load(self.source_files[index])


class TensorDataset(Dataset):
    def __init__(self, source, device='cpu'):
        self.data = torch.load(source)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]
