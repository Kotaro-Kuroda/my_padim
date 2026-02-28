import glob
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2


class MVTecDataset(Dataset):
    def __init__(self, root_dir, categories, input_size, pad_size=0, random_affine=False, sub_types=None):
        self.root_dir = root_dir
        self.categories = categories
        size = input_size - pad_size
        transform_list = [
            transforms.ToTensor(),
            transforms.Resize((size, size)),
            transforms.Pad(pad_size // 2),
        ]
        if random_affine:
            transform_list.append(transforms.RandomAffine(degrees=3, translate=(0.01, 0.01)))
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(
            transform_list
        )
        self.data = []

        for category in categories:
            listdir = os.listdir(os.path.join(root_dir, category, 'train'))
            for dir_name in listdir:
                if sub_types is not None and dir_name not in sub_types:
                    continue

                category_dir = os.path.join(root_dir, category, 'train', dir_name)
                images = glob.glob(os.path.join(category_dir, '*.png')) + glob.glob(os.path.join(category_dir, '*.jpg')) + glob.glob(os.path.join(category_dir, '*.jpeg'))
                samples = 5000 if sub_types is None else len(images)
                indices = np.random.choice(len(images), size=min(samples, len(images)), replace=False)
                for i in range(len(images)):
                    if i in indices:
                        self.data.append(images[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = Image.open(image).convert('RGB')
        image = self.transform(image)
        return image
