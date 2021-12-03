from sklearn.metrics import precision_score, recall_score, f1_score
from .variables import getVariableGroups
diagnosis_variables, area_variables, characteristics_variables = getVariableGroups()
type_default = f1_score
average_default = 'macro'

def compute_matrix_metrics(pred, target, average, type, single_scores):
    # Check size
    assert len(pred) == len(target)
    # Calculate total score
    total_score = type(y_true=target, y_pred=pred, average=average, zero_division=0)
    # if single_char_scores is True, calculate individual scores and return dict of total and singles combined
    if single_scores:
        all_scores = {}
        all_scores['total'] = total_score
        all_scores['singles'] = {}
        for col in pred.columns:
            all_scores['singles'][col] = compute_metrics(pred[col], target[col], average, type)
        return all_scores
    else:
        return total_score
    
def compute_metrics(pred, target, average, type):
    # Check size
    assert len(pred) == len(target)
    # Calculate score. NB: Check for dimension and average type
    if pred.ndim == 1 and average == 'samples':
        average = average_default
    return type(y_true=target, y_pred=pred, average=average, zero_division=0)

def compute_metrics_scores(pred, target, average=average_default, type=type_default, single_char_scores=False, single_area_scores=False):
    assert pred.shape == target.shape

    # Compute characteristics metric score
    characteristics_scores = {'characteristics': compute_matrix_metrics(
        pred[characteristics_variables],
        target[characteristics_variables],
        average,
        type,
        single_char_scores)}

    # Compute diagnosis metric score
    diagnosis_scores = { 'diagnosis': compute_metrics(
        pred[diagnosis_variables],
        target[diagnosis_variables],
        average,
        type)}

    # Compute area accuracy metric score
    area_scores = {'area': compute_matrix_metrics(
        pred[area_variables],
        target[area_variables],
        average,
        type,
        single_area_scores)}
    
    return characteristics_scores, diagnosis_scores, area_scores