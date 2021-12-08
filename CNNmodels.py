#Dependencies
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self, n_characteristics = 7, n_diagnosis = 6, n_area = 4):
        super().__init__()
        
        #Take the efficientnet_b0 
        model = models.googlenet(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1024) #Replace the last layer.
        self.base_model = model 

        self.fc_characteristics = nn.Linear(in_features = 1024, out_features = n_characteristics)
        self.fc_diagnosis       = nn.Linear(in_features = 1024, out_features = n_diagnosis)
        self.fc_area            = nn.Linear(in_features = 1024, out_features = n_area)

    def forward(self, x):
        #Pass the image through resnet. It's still being trained.
        x = self.base_model(x) 
        x = torch.nn.ReLU()(x)

        #Characteristics pipeline
        x_characteristics = self.fc_characteristics(x)
        x_characteristics = torch.sigmoid(x_characteristics)

        #Diagnosis pipeline
        x_diagnosis = self.fc_diagnosis(x)
        x_diagnosis = torch.softmax(x_diagnosis, dim=-1) #Since it should sum to 1.

        #Area pipeline
        x_area = self.fc_area(x)
        x_area = torch.softmax(x_area, dim=-1)

        #Concatenate the networks together again.
        x_cat  = torch.cat((x_characteristics, x_diagnosis, x_area), dim=1)
                
        return x_cat