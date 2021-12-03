from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms.functional as TF
import numpy as np
import torch

class CharacteristicsDataset(Dataset):
    def __init__(self, path, target, size = None, transform = None):
        self.path   = path
        self.target = torch.from_numpy(target.values)
        self.variables = target.columns.tolist()
        self.transform = transform
        self.size = size
        
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        #Fetch the path and image
        path  = self.path[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #Fetch the target
        target = self.target[idx,:]
        
        # Resize
        if self.size is not None:
            image = cv2.resize(image, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        
        # Transform
        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        # Cast to Tensor
        image = TF.to_tensor(image)
        
        return image, target, self.variables
