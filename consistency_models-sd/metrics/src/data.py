import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _filename(path):
    return os.path.basename(path).split('.')[0]


def numerical_order(files):
    return sorted(files, key=lambda x: int(x.split('.')[0]))


class UnannotatedDataset(Dataset):
    def __init__(self, root_dir, numerical_sort=False, transform=None):
        self.img_files = []
        for root, _, files in os.walk(root_dir):
            for file in numerical_order(files) if numerical_sort else sorted(files):
                if UnannotatedDataset.file_is_img(file):
                    self.img_files.append(os.path.join(root, file))
        self.transform = transform

    @staticmethod
    def file_is_img(name):
        extension = os.path.basename(name).split('.')[-1]
        return extension in ['jpg', 'jpeg', 'png', 'webp', 'JPEG']

    def align_names(self, target_names):
        new_img_files = []
        img_files_names_dict = {_filename(f): f for f in self.img_files}
        for name in target_names:
            try:
                new_img_files.append(img_files_names_dict[_filename(name)])
            except KeyError:
                print('names mismatch: absent {}'.format(_filename(name)))
        self.img_files = new_img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        img = Image.open(self.img_files[item])
        img = img.convert('RGB')
        if self.transform is not None:
            return self.transform(img)
        else:
            return img
        

def get_dinov2_loader(path, batch_size=256):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_DEFAULT_MEAN,
            IMAGENET_DEFAULT_STD
        )
    ])
    dataset = UnannotatedDataset(path, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=False, num_workers=0
    )
    return loader