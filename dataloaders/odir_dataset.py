import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from dataloaders.data_utils import get_unk_mask_indices, image_loader
from torchvision import transforms


class OdirDataset(Dataset):

    def __init__(self, split, num_labels, data_file, transform=None, known_labels = 0, testing=False):

        self.split = split
        self.split_data = np.load(data_file, allow_pickle=True)
        self.root = "/kaggle/input/oia-odir-5k/oia-odir"

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
        self.num_labels = num_labels
        self.known_labels = known_labels
        self.testing = testing

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_ID = self.imgs[idx]

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

        # image = image[0].unsqueeze(0)
        # print(image.shape)

        unk_mask_indices = get_unk_mask_indices(image,self.testing,self.num_labels,self.known_labels)
        
        mask = labels.clone()
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)

        sample = {}
        sample['image'] = image
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = image_ID
        return sample