from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from .variables import getVariableGroups
diagnosis_variables, area_variables, characteristics_variables = getVariableGroups()
type_default = f1_score
average_defaul = 'samples'
average_secondary = 'macro'

def compute_matrix_metrics(pred, target, average, type, single_char_scores, single_char):
    # Check size
    assert len(pred) == len(target)
    # Calculate total score
    total_score = type(y_true=target, y_pred=pred, average=average, zero_division=0)
    # if single_char_scores is True, calculate individual scores and return dict of total and singles combined
    if single_char_scores:
        all_scores = {}
        all_scores['total'] = total_score
        all_scores['singles'] = {}
        for col in pred.columns:
            all_scores['singles'][col] = compute_accuracy_metrics(pred[col], target[col], average, type, single_char)
        return all_scores
    else:
        return total_score
    
def compute_accuracy_metrics(pred, target, average, type, single_acc):
    # Check size
    assert len(pred) == len(target)
    # Calculate score. NB: Check for single or multidimensional
    if pred.ndim == 1 and average == 'samples':
        average = average_secondary
    score = type(y_true=target, y_pred=pred, average=average, zero_division=0)
    # If single_acc is True, also return accuracy
    if single_acc:
        return { 'score': score, 'accuracy': accuracy_score(pred, target) }
    else:
        return score

def compute_metrics_scores(pred, target, average=average_defaul, type=type_default, single_char_scores=False, single_acc=False):
    assert pred.shape == target.shape

    # Compute characteristics metric score
    characteristics_scores = compute_matrix_metrics(
        pred[characteristics_variables],
        target[characteristics_variables],
        average,
        type,
        single_char_scores,
        single_acc)

    # Compute diagnosis metric score
    diagnosis_scores = compute_accuracy_metrics(
        pred[diagnosis_variables],
        target[diagnosis_variables],
        average,
        type,
        single_acc)

    # Compute area accuracy metric score
    area_scores = compute_accuracy_metrics(
        pred[area_variables],
        target[area_variables],
        average,
        type,
        single_acc)
    
    return characteristics_scores, diagnosis_scores, area_scores