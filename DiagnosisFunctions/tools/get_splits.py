import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def get_splits(random_state = 42, data_folder = './data/', file_extension = '.jpeg'):
    
    images_path = os.path.join(data_folder, 'images')
    labels_path = os.path.join(data_folder, 'dermx_labels.csv')
    
    labels = pd.read_csv(labels_path)

    labels['path']   = labels['image_id'].apply(lambda x : os.path.join(images_path, f'{x}.jpeg'))
    labels['exists'] = labels['path'].apply(lambda x : os.path.exists(x)).astype(int)

    #Drop the rows we don't have data on.
    labels = labels.loc[labels['exists'] == 1]
    
    le = LabelEncoder()
    
    #To begin with, we just want the path and the diagnosis.
    data = pd.DataFrame({'path' : labels['path'], 'target' : le.fit_transform(labels['diagnosis'])})
    
    
    #Train is 70%, val is 15% and test is 15%. 
    train, val = train_test_split(data, test_size = 0.3, random_state = random_state)
    val, test  = train_test_split(val, test_size = 0.5, random_state = random_state)
    
    
    
    return (train.path.tolist(), train.target.tolist()), (val.path.tolist(),   val.target.tolist()), (test.path.tolist(),  test.target.tolist()), le

def get_splits_characteristics(random_state = 42, data_folder = './data/', file_extension = '.jpeg'):
    
    images_path = os.path.join(data_folder, 'images')
    labels_path = os.path.join(data_folder, 'dermx_labels.csv')
    
    labels = pd.read_csv(labels_path)

    labels['path']   = labels['image_id'].apply(lambda x : os.path.join(images_path, f'{x}.jpeg'))
    labels['exists'] = labels['path'].apply(lambda x : os.path.exists(x)).astype(int)

    #Drop the rows we don't have data on.
    labels = labels.loc[labels['exists'] == 1]
    
    le = LabelEncoder()

    data = labels.filter(['path', 'scale', 'plaque', 'pustule', 'patch', 'papule', 'dermatoglyph_disruption', 'open_comedo'], axis=1)
    data['area'] = le.fit_transform(labels['diagnosis'])
        
    #Train is 70%, val is 15% and test is 15%. 
    train, val = train_test_split(data, test_size = 0.3, random_state = random_state)
    val, test  = train_test_split(val, test_size = 0.5, random_state = random_state)
    
    
    return (train.path.tolist(), train.drop('path', axis=1)), (val.path.tolist(), val.drop('path', axis=1)), (test.path.tolist(),  test.drop('path', axis=1)), le