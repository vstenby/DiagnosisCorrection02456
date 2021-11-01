import os
import numpy as np
from sklearn.model_selection import train_test_split

def get_splits(random_state = 42, images_folder = './data/images/', file_extension = '.jpeg'):
    #Get all images
    images = np.array([os.path.join('./data/images/', x) for x in os.listdir('./data/images') if x.endswith('.jpeg')])
    
    #Train is 70%, val is 15% and test is 15%. 
    train, val = train_test_split(images, test_size = 0.3, random_state = random_state)
    val, test  = train_test_split(val, test_size = 0.5,    random_state = random_state)
    
    return train, val, test