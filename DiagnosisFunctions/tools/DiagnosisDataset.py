from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms.functional as TF
import numpy as np

def center_crop(img, new_width=None, new_height=None):        

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img

class DiagnosisDataset(Dataset):
    def __init__(self, path, target, size = None, transform = None):
        self.path   = path
        self.target = target
        self.transform = transform
        self.center_crop = center_crop
        self.size = size
        
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        #Fetch the path and image
        path  = self.path[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #Fetch the target
        target = self.target[idx]
        
        if self.transform is not None:
            image = self.transform(image=image)['image']
            
        image = TF.to_tensor(image)
        
        #Crop to center afterwards.
        image = TF.center_crop(image, output_size = min(image.shape[1:]))
        
        if self.size is not None:
            image = TF.resize(image, size = self.size)
        
        return image, target