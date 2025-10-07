import os
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy as np
from typing import Tuple,Union


class custom_Dataset(Dataset):
    def __init__(self,root,train=True,transform = None):
        self.root = root
        self.train = train
        self.transform = transform

        if train:
            self.images_path = os.path.join(root, 'train-images-idx3-ubyte')
            self.labels_path = os.path.join(root, 'train-labels-idx1-ubyte')
        else:
            self.images_path = os.path.join(root, 't10k-images-idx3-ubyte')
            self.labels_path = os.path.join(root, 't10k-labels-idx1-ubyte')

        self.images , self.labels = self._load_data()
    def _load_data(self) -> Tuple[np.ndarray,np.ndarray]:
        with open(self.images_path,'rb') as f:
            magic = int.from_bytes(f.read(4),byteorder= 'big')
            num_images = int.from_bytes(f.read(4),byteorder='big')
            rows = int.from_bytes(f.read(4),byteorder='big')
            cols = int.from_bytes(f.read(4),byteorder='big')
            images = np.frombuffer(f.read(),dtype=np.uint8).reshape(num_images,rows,cols)
        with open(self.labels_path,'rb') as f:
            magic = int.from_bytes(f.read(4),byteorder='big')
            num_labels = int.from_bytes(f.read(4),byteorder='big')
            labels = np.frombuffer(f.read(),dtype=np.uint8)

        return images,labels
    def __len__(self) -> int:
        return len(self.images)
    def __getitem__(self, index:int) -> Tuple[torch.Tensor,torch.Tensor]:
        image = self.images[index]
        label = self.labels[index]

        image = Image.fromarray(image,mode='L')
        if self.transform:
            image = self.transform(image)
        else:
            default_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))
            ])
            image = default_transform(image)
        return image, torch.tensor(label, dtype=torch.long)
    
def transform(image:Image.Image) -> torch.Tensor:
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ]
    )
    return preprocess(image)

train_dataset = custom_Dataset(
    root="./data/MNIST/raw/",
    train = True,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

