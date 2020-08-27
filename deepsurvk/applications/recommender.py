# -*- coding: utf-8 -*-
"""
recommender.py
Functions for the treatment recommender.
"""
import numpy as np
#from lifelines.utils import concordance_index as c_index

__all__ = ['recommender_function',
           'get_recs_antirecs_index']

#%%
def recommender_function(model, X, treatment_column):
    """
    Calculate the recommender function for a set of patients based on a 
    previously fitted model. Implementation corresponds to the one 
    proposed in [1] (Eq. 6).
    
    Parameters
    ----------
    model: 
        Model that will be used to compute the log-hazards.
        It needs to be fitted previously.
    X: pandas DataFrame
        Data. Rows correspond to instances (i.e, patients).
        Columns correspond to features.
    treatment_column: string
        Which column in X corresponds to the treatment.

    Currently, it only supports comparison of two treatments.

    Returns
    -------
    rec_ij: NumPy array.
        The recommender function of all patients. For each patient:
        If rec_ij is positive, it means that treatment i lead to a higher 
        risk of death than treatment j.
        If rec_ij is negative, it means that treatment j lead to a higher
        risk of death than treatment i.
        
    References
    ----------
    [1] Katzman, Jared L., et al. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." BMC medical research methodology 18.1 (2018): 24.
    """
    
    # Validate number of treatments.
    treatments = X[treatment_column].unique()
    n_treatments = len(treatments)
    
    if n_treatments == 1:
        raise ValueError("It is not possible to give a treatment recommendation with only one treatment value.")
    elif n_treatments > 2:
        raise ValueError(f"{n_treatments} found. Currently, only two treatments are supported for comparison.")
    
    # Create DataFrames to be used for prediction.
    X_treatment0 = X.copy(deep=True)
    X_treatment0[treatment_column] = treatments[0]
    
    X_treatment1 = X.copy(deep=True)
    X_treatment1[treatment_column] = treatments[1]
    
    # Calculate the log-hazards.
    h_i = model.predict(X_treatment0)
    h_j = model.predict(X_treatment1)
    
    # Calculate the recommender function (Eq. 6)
    rec_ij = h_i - h_j
    
    return rec_ij


#%%
def get_recs_antirecs_index(rec_ij, X, treatment_column):
    """
    Get the indexes of the recommendation patients (patients that were 
    treated according to the model's suggested treatment) and of the
    antirecommendation patients (patients that were NOT treated according
    to the model's suggested treatment). Implementation corresponds to the one 
    proposed in [1].
    
    Parameters
    ----------
    rec_ij: NumPy array
        Recommender function as given by recommender_function.
    X: pandas DataFrame
        Data. Rows correspond to instances (i.e, patients).
        Columns correspond to features.        
    treatment_column: string
        Which column in X corresponds to the treatment.

    Returns
    -------
    tuple
        First element, recommendation_idx, corresponds to a boolean array
        with value of True of the recommendation patients. 
        Second element, antirecommendation_idx, corresponds to a boolean
        array with value of True for the antirecommendation patients.
        This output is given for completeness, since it is simply the negated 
        version of recommendation_idx.
        
    References
    ----------
    [1] Katzman, Jared L., et al. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." BMC medical research methodology 18.1 (2018): 24.
    """
    
    # Convert rec_ij into the actual recommended treatment (binary array).
    # If rec_ij was positive, that means that treatment i (i.e., 0)
    # had a higher risk. Thus, treatment j (i.e., 1) was better.
    # If rec_ij was negative, that means that treatment j (i.e., 1)
    # had a higher risk. Thus, treatment i (i.e., 0) was better.
    recommended_treatment = (rec_ij > 0).astype(np.int32)
    
    # Get groups of patients that were and were not treated according to the 
    # model's recommended treatment. Following the paper's nomenclature, 
    # these will be recommendation and antirecommendation, respectively.
    real_treatment = X[treatment_column].values
    
    recommendation_idx = np.logical_and(recommended_treatment.reshape((-1,)), real_treatment.reshape((-1,)))
    antirecommendation_idx = ~recommendation_idx
    
    return recommendation_idx, antirecommendation_idx