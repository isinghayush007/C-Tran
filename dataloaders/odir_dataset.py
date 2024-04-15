import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from dataloaders.data_utils import get_unk_mask_indices, image_loader
from torchvision import transforms


class OdirDataset(Dataset):

    def __init__(self, split, data_file, transform=None):

        self.split = split
        self.split_data = np.load(data_file, allow_pickle=True)
        self.root = "/kaggle/input/oia-odir"

        if self.split == 'train':
            self.imgs = self.split_data['train_images']
            self.labs = self.split_data['train_labels']
        elif self.split == 'val':
            self.imgs = self.split_data['val_images']
            self.labs = self.split_data['val_labels']
        elif self.split == 'test':
            self.imgs = self.split_data['test_images']
            self.labs = self.split_data['test_labels']
        else:
            raise ValueError

        self.transform = transform
        self.resizer = transforms.Resize((640, 640))

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        labels = self.labs[idx].astype(int)
        labels = torch.Tensor(labels)
        
        image = Image.open(self.root + "/" + self.imgs[idx])
        image = torch.Tensor(np.array(image))
        image = image.permute(2, 0, 1)
        
        if self.transform is not None:
            image = self.transform(image)
        # if len(image.shape) > 2:
        #     # image = image[:, :, 0]
        #     image = image[0]

        image = image[0].unsqueeze(0)
        # print(image.shape)

        sample = {}
        sample['image'] = image
        sample['labels'] = labels

        return sample



