import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def add_drift(curr: pd.Series, drift_size: float) -> pd.Series:
    """Artificially adds a shift to the data.
    Args:
        curr: initial data
        drift_size: percent initial values would be increased by
    Returns:
        curr: drifted data
    """
    
    alpha = 0.001
  
    delta = (alpha + np.mean(curr)) * drift_size
    new = curr + delta

    return new


def plot_auc_drift(drift_features: list, ohe_test_data: pd.DataFrame, drift_sizes = np.linspace(0, 1, 51), drift_direction = 'positive',
                   drift_score_plot = True, y_lim = 'default'):
    """Artificially drifts features by increasing increments and plots the AUC at each.
    Args:
        drift_features: list of features to drift
        ohe_test_data: one-hot-encoded test data
        drift_sizes: list of amounts of drift to test
        drift_direction: 'postive', 'negative', or 'random'. Indicates direction of feature drift. 'random' applies a random direction
                          to each feature.
        drift_score_plot: True or False. If True also plots Wasserstein normed distance drift score against drift percentage as separate plot.
    Returns:
        plot of drift percentage vs ROC_AUC
    """
    
    # Initialise storage lists/dictionaries and data
    auc_list = []
    drifted_ohe_test_data = ohe_test_data.copy()
    coeff = np.ones(len(drift_features))
    
    drift_scores_dict = {k: None for k in drift_sizes}
    
    # # Create the coefficients dependent on type of drift
    # if drift_direction == 'negative':
    #     coeff = -1*coeff
    # elif drift_direction == 'random':
    #     func = np.vectorize(lambda x: 1 if x*random.random() < 0.5 else -1)
    #     coeff = func(coeff)
    
    # Calculate ROC_AUC and drift score for each feature at each drift size
    for drift in drift_sizes:
        for i, feature in enumerate(drift_features):
            drifted_ohe_test_data[feature] = add_drift(curr = ohe_test_data[feature], drift_size = coeff[i]*drift)
            drift_scores_dict[feature][drift] = stattest(ohe_test_data[feature], drifted_ohe_test_data[feature], 'num', 0.1)[0]
            
        output = model.predict_proba(drifted_ohe_test_data)[:,1]
        fpr, tpr, thresholds = roc_curve(y_true=target, y_score=output)

        auc_list.append(auc(fpr, tpr))
    
    # Print each feature and how they were drifted
    print('Drifted features:')
    for i, feature in enumerate(drift_features):
        if coeff[i] == 1: 
            print(feature, ' (postive drift),')
        else:
            print(feature, ' (negative drift),')
    
    # Plot ROC_AUC vs drift percentage
    f = plt.figure()
    plt.locator_params(axis='x', nbins=20)
    plt.title(f"ROC_AUC at different levels of {drift_direction} feature drift")
    plt.xlabel('Drift Percentage')
    plt.ylabel('ROC_AUC')
    
    if y_lim != 'default':
        plt.ylim(y_lim)
    
    plt.plot(drift_sizes*100, auc_list)