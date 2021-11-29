#Dependencies
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

class CNN1(nn.Module):
    def __init__(self, n_characteristics = 7, n_diagnosis = 6, n_area = 4):
        super().__init__()
    
        #We can still fine tune it.
        self.base_model = models.resnext50_32x4d(pretrained=True)

        self.fc_diagnosis       = nn.Linear(in_features = self.base_model.fc.out_features, out_features = n_diagnosis)

    def forward(self, x):
        #Pass the image through resnet. It's still being trained.
        x = self.base_model(x) 

        #Diagnosis pipeline
        x_diagnosis = self.fc_diagnosis(x)
        x_diagnosis = torch.softmax(x_diagnosis, dim=-1) #Since it should sum to 1.
        
        return x_diagnosis

class CNN2(nn.Module):
    def __init__(self, n_characteristics = 7, n_diagnosis = 6, n_area = 4):
        super().__init__()
        
        #We can still fine tune it.
        self.base_model = models.resnext50_32x4d(pretrained=True)

        self.fc_characteristics = nn.Linear(in_features = self.base_model.fc.out_features, out_features = n_characteristics)
        self.fc_diagnosis       = nn.Linear(in_features = self.base_model.fc.out_features, out_features = n_diagnosis)
        self.fc_area            = nn.Linear(in_features = self.base_model.fc.out_features, out_features = n_area)

    def forward(self, x):
        #Pass the image through resnet. It's still being trained.
        x = self.base_model(x) 

        #Characteristics pipeline
        x_characteristics = self.fc_characteristics(x)
        x_characteristics = torch.sigmoid(x_characteristics)

        #Diagnosis pipeline
        x_diagnosis = self.fc_diagnosis(x)
        x_diagnosis = torch.softmax(x_diagnosis, dim=-1) #Since it should sum to 1.

        #Area pipeline
        x_area = self.fc_area(x)
        x_area = torch.softmax(x_area, dim=-1)

        x_cat  = torch.cat((x_characteristics, x_diagnosis, x_area), dim=1)
        
        return x_cat