from PIL import Image
import torch
from torch.utils.data import Dataset
import logging
import torch
from torch.utils import data
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import os
from PIL import Image
import numpy as np
import albumentations as alb
import random
from torch.utils.data.dataloader import default_collate


class MyDataSet(Dataset):

    def __init__(self, root1, root2):
        face = os.listdir(root1)
        self.face = [os.path.join(root1, k) for k in face]
        voice = os.listdir(root2)
        self.voice = [os.path.join(root2, k) for k in voice]

    def __getitem__(self, index):
        face = self.face[index]
        voice = self.voice[index]
        label = face[-8]
        if label == 'D':
            label = 1
        else:
            label = 0

        face_data = torch.from_numpy(np.load(face, allow_pickle=True).astype(float)).float()
        voice_data = torch.from_numpy(np.load(voice, allow_pickle=True).astype(float)).float()

        return face_data[:, :, 0: 24], voice_data[0: 24, :], label

    def __len__(self):
        return len(self.face)

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range


if __name__ == "__main__":
    path1 = r'D:\Codedemo\dep-mul\mul\data\face'
    path2 = r'D:\Codedemo\dep-mul\mul\data\voic'
    print(MyDataSet(path1, path2))
