import numpy as np
import pandas as pd
from .variables import getVariableGroups
diagnosis_variables, area_variables, characteristics_variables = getVariableGroups()
threshold = 0.5

def classify_multiple_labels(pred_prob, variables):
    # Translate all probabilities to classifiations based on threshold
    for var in variables:
        pred_prob[var] = (pred_prob[var] >= threshold).astype('int')

    return pred_prob[variables]

def classify_single_label(pred_prob, variables):
    # Initilize binary array
    labels_binary = np.zeros((pred_prob[variables].shape))
    # Find most likely type for all indicies and set to 1
    pred_labels = pred_prob[variables].values.argmax(axis=1)
    for idx, label in enumerate(pred_labels):
        labels_binary[idx,label] = 1
    # Update variable columns
    pred_prob.loc[:,variables] = labels_binary.astype('int')

    return pred_prob[variables]

def classify_probability_predictions(pred):

    # Predict characteristics variables
    pred[characteristics_variables] = classify_multiple_labels(pred, characteristics_variables)

    # Predict diagnosis variables
    pred[diagnosis_variables] = classify_single_label(pred, diagnosis_variables)

    # Predict area variables
    pred[area_variables] = classify_single_label(pred, area_variables)

    # Return classified predictions
    return pred