from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms.functional as TF
import numpy as np

class CharacteristicsDataset(Dataset):
    def __init__(self, path, target, size = None, transform = None):
        self.path   = path
        self.scale = target['scale'].tolist()
        self.plaque = target['plaque'].tolist()
        self.pustule = target['pustule'].tolist()
        self.patch = target['patch'].tolist()
        self.papule = target['papule'].tolist()
        self.dermatoglyph_disruption = target['dermatoglyph_disruption'].tolist()
        self.open_comedo = target['open_comedo'].tolist()
        self.area = target['area'].tolist()
        self.transform = transform
        self.size = size
        
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        #Fetch the path and image
        path  = self.path[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #Fetch the targets
        scale = self.scale[idx]
        plaque = self.plaque[idx]
        pustule = self.pustule[idx]
        patch = self.patch[idx]
        papule = self.papule[idx]
        dermatoglyph_disruption = self.dermatoglyph_disruption[idx]
        open_comedo = self.open_comedo[idx]
        area = self.area[idx]
        
        if self.transform is not None:
            image = self.transform(image=image)['image']
            
        image = TF.to_tensor(image)
        
        #Crop to center afterwards.
        #image = TF.center_crop(image, output_size = min(image.shape[1:]))
        
        if self.size is not None:
            image = TF.resize(image, size = self.size)
        
        return image, [scale, plaque, pustule, patch, papule, dermatoglyph_disruption, open_comedo, area]
