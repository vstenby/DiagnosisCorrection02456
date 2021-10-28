import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import tqdm.auto as tqdm

import os
from PIL import Image
from sklearn.metrics import accuracy_score
import torchvision
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

import argparse

class DiagnosisDataset(Dataset):
    '''
    Define our dataset
    '''
    def __init__(self, path, target, transforms = torch.nn.Sequential()):
        #Input:
        # path:   path to the images.
        # target: target diagnosis.
        
        assert len(path) == len(target), 'path and target should be the same length.'
        
        self.path   = path
        self.target = target
        self.transforms = transforms
        
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        path   = self.path[idx]
        target = self.target[idx]
        
        #Load the image
        im = Image.open(path)
        im = np.array(im) #4th channel is alpha.
        im = torch.tensor(im, dtype=torch.float32).permute(2,0,1) / 255.
        
        if self.transforms is not None:
            im = self.transforms(im)
            
        return im, target, path
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequential = nn.Sequential(
                              nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
                              nn.BatchNorm2d(num_features=32),
                              nn.ReLU(),
            
                              #nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
                              #nn.BatchNorm2d(num_features=32),
                              #nn.ReLU(),
            
                              nn.MaxPool2d(kernel_size=2,stride=2),
            
                              nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
                              nn.BatchNorm2d(num_features=64),
                              nn.ReLU(),
                            
                              #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
                              #nn.BatchNorm2d(num_features=64),
                              #nn.ReLU(),
                        
                              nn.MaxPool2d(kernel_size=2,stride=2),
                            
                              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
                              nn.BatchNorm2d(num_features=128),
                              nn.ReLU(),
                            
                              nn.MaxPool2d(kernel_size=2,stride=2),
                            
                              nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
                              nn.BatchNorm2d(num_features=256),
                              nn.ReLU(),
            
                              #nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                              #nn.BatchNorm2d(num_features=128),
                              #nn.ReLU(),
            
                              #nn.MaxPool2d(kernel_size=2,stride=2),
            
                              #nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                              #nn.BatchNorm2d(num_features=32),
                              #nn.ReLU()
            
                            )    
        
        self.fc = nn.Linear(in_features = 256*8*8, out_features = 6)
                            
        
    def forward(self, x):
        x = self.sequential(x)
        x = x.reshape(-1, 256*8*8)
        x = self.fc(x)
        #No activation since we're using CrossEntropyLoss().
        return x
    
def main():
    
    #Parse the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-epochs', default=5, type=int,               help='Number of epochs we should train.')
    parser.add_argument('--csv-path', default='results.csv', type=str, help='Path for the results.csv file')
    args = parser.parse_args()
    
    #Set the number of epochs.
    num_epochs = args.n_epochs
    
    #Set up the device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Make sure we have all of the files.
    files = ['diseases_characteristics.csv', 'dermx_labels.csv']

    for file in files:
        assert file in os.listdir('./data/'), f'{file} should be in the ./data/ folder'
    
    dermx_labels = pd.read_csv('./data/dermx_labels.csv')
    dermx_labels.groupby('diagnosis').size()

    #Transform the labels.
    le = LabelEncoder().fit(dermx_labels['diagnosis'].tolist())
    dermx_labels['target'] = le.transform(dermx_labels['diagnosis'])

    #Write up the image path.
    dermx_labels['path'] = dermx_labels['image_id'].apply(lambda x : os.path.join('data', 'images', f'{x}.jpeg'))

    #Drop images that could not be found.
    n_before = len(dermx_labels)

    dermx_labels = dermx_labels.loc[dermx_labels['path'].apply(lambda x : os.path.exists(x))]

    n_after  = len(dermx_labels)

    print(f'Observations before: {n_before}')
    print(f'Observations after: {n_after}')

    assert n_after > 0, 'Images could not be found. Make sure you have unpacked images into ./data/images/'
    
    
    #Let's just set train to first 0-100, val to 100-200 and test to 200-300.
    train = dermx_labels.iloc[:100,:]
    val   = dermx_labels.iloc[100:200, :]
    test  = dermx_labels.iloc[200:300, :]
    
    #Set the transform
    transforms = torch.nn.Sequential(torchvision.transforms.CenterCrop(100))

    train_dataset   = DiagnosisDataset(train['path'].tolist(), train['target'].tolist(), transforms=transforms)
    train_loader    = DataLoader(train_dataset, batch_size = 16, shuffle=True)  

    val_dataset     = DiagnosisDataset(val['path'].tolist(), val['target'].tolist(), transforms=transforms)
    val_loader      = DataLoader(val_dataset, batch_size = 16, shuffle=False)  

    test_dataset    = DiagnosisDataset(test['path'].tolist(), test['target'].tolist(), transforms=transforms)
    test_loader     = DataLoader(test_dataset, batch_size = 16, shuffle=False)  

    cnn = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters())

    train_losses     = []
    train_accuracies = []

    val_losses       = []
    val_accuracies   = []


    for epoch in tqdm.tqdm(range(num_epochs), unit='epoch', desc="Epoch"):

        #Epochs start @ 1 now.
        epoch += 1


        ## -- Training -- ##
        cnn.train()
        train_loss = 0
        predictions  = []
        ground_truth = []

        for data in train_loader:

            #Fetch images and targets from train loader.
            images, targets, _ = data

            #Get that stuff on the GPU
            images  = images.to(device)
            targets = targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(images)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            #Add the batch loss
            train_loss += loss.item()

            #Save predictions and targets
            predictions  += outputs.argmax(axis=1).tolist()
            ground_truth += targets.tolist()

        #Append this epoch's statistics.
        train_losses.append(train_loss)
        train_accuracies.append(accuracy_score(ground_truth, predictions))

        ## -- End of Training -- ##

        ## -- Validation -- ##
        cnn.eval()
        val_loss = 0
        predictions  = []
        ground_truth = []

        for data in val_loader:

            #Fetch images and targets from val loader.
            images, targets, _ = data

            #Get that stuff on the GPU
            images  = images.to(device)
            targets = targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(images)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            #Add the batch loss
            val_loss += loss.item()

            #Save predictions and targets
            predictions  += outputs.argmax(axis=1).tolist()
            ground_truth += targets.tolist()

        #Append this epoch's statistics.
        val_losses.append(val_loss)
        val_accuracies.append(accuracy_score(ground_truth, predictions))
        ## -- End of Validation -- ##
        
    
    #Save to csv file.
    pd.DataFrame({'Epoch' : np.arange(1, num_epochs+1),
                  'Train Loss' : train_losses,
                  'Train acc'  : train_accuracies,
                  'Val Loss'   : val_losses,
                  'Val acc'    : val_accuracies}).to_csv(args.csv_path, index=False)
                  
    
    return

if __name__ == '__main__':
    main()