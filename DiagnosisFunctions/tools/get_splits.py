import os
import numpy as np
from sklearn.model_selection import train_test_split

def get_splits(random_state = 42, data_folder = './data/', file_extension = '.jpeg'):
    
    images_path = os.path.join(data_folder, 'images')
    labels_path = os.path.join(data_folder, 'dermx_labels.csv')
    
    #Get all images
    images = np.array([os.path.join(images_path, x) for x in os.listdir(images_path) if x.endswith('.jpeg')])
    
    #Load the labels
    labels = pd.read_csv(labels_path)
    
    #Train is 70%, val is 15% and test is 15%. 
    train, val = train_test_split(labels, test_size = 0.3, random_state = random_state)
    val, test  = train_test_split(val, test_size = 0.5, random_state = random_state)
    
    
    
    return train, val, test