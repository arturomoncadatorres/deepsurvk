# -*- coding: utf-8 -*-
"""
recommender.py
Functions for the treatment recommender.
"""

#from lifelines.utils import concordance_index as c_index

__all__ = ['recommend_treatment']

#%%
def recommend_treatment(model, X, treatment_column):
    """
    Recommend a treatment for a patient based on a previously fitted model.
    Implementation corresponds to the one proposed in [1].
    
    Parameters
    ----------
    model: 
        Model that will be used to recommend a treatment.
        It needs to be fitted previously.
        
    X: NumPy array or pandas DataFrame
        Data. Rows correspond to instances (i.e, patients).
        Columns correspond to features.
        
    treatment_column: int or string
        Indicator of which column in X corresponds to the treatment.
        If X is a NumPy array, treatment_column must be an int.
        If X is a pandas DataFrame, treatment_column must be a string.

    Currently, it only supports comparison of two treatments.            

    Returns
    -------
    recommendation: 
        The recommended treatment for each instance (i.e., patient).
        
    References
    ----------
    [1] Katzman, Jared L., et al. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." BMC medical research methodology 18.1 (2018): 24.
    """
    
    c = c_index(y_true, y_pred, E)
    return c