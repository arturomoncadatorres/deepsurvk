# -*- coding: utf-8 -*-
"""
optimization.py
Functions for (hyper)parameter optimization.
"""
import datetime
import numpy as np
import itertools
from sklearn.model_selection import RepeatedKFold

import deepsurvk

# import logzero
# from logzero import logger

__all__ = ['optimize_hp']


#%%
def optimize_hp(X, Y, E, mode='grid', n_splits=3, n_repeats=5, verbose=True, **params):
    """
    Optimize the (hyper)parameters of a DeepSurvK model using 
    cross-validation.
    
    Parameters
    ----------
    X: pandas DataFrame
        Data
    Y: pandas DataFrame
        It needs to have column 'T'
    E: pandas DataFrame
        It needs to have column 'E'
    mode: string
        Possible values are:
            'grid' (default)
            'random' TODO
    n_splits: int (optional)
        Number of folds. Default value is 3, as suggested in [1].
    n_repeats: int (optional)
        Number of CV repetition. Default value is 5.
    verbose: boolean (optional)
        Define if verbose output is desired (True, default) or not (False)
    params: dictionary
        Each key corresponds to a parameter. 
        The values correspond to a list of parameters to be explored.
        
        The number of epochs can be defined here. It should also be given as
        an entry of the dictionary with key `epochs` and value a list
        comprised of only one element. If the list has more than
        one element, only the first one will be considered. If number
        of epochs isn't defined by the user, then a default of 1000 will
        be used.
            
    Returns
    -------
    best_params: dictionary
        Best parameters.
        Each key corresponds to a parameter. 
        The values correspond to the optimized parameter.
        
    References
    ----------
    [1] Katzman, Jared L., et al. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." BMC medical research methodology 18.1 (2018): 24.
    """
    
    # Check if number of epochs was defined.
    if 'epochs' in params:
        # If yes, extract its value (and remove it from the dictionary,
        # since it won't be optimized).
        epochs = params['epochs'][0]
        params.pop('epochs')
        
    else:
        # If not, set a default value of 1000.
        epochs = 1000
        
    
    # Generating a list of dictionaries with all possible combinations.
    # Trick from https://stackoverflow.com/a/61335465/948768
    keys, values = zip(*params.items())
    params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Compute important parameters.
    n_features = X.shape[1]
    n_combinations = len(params_list)
    if verbose:
        print(f"Optimizing {n_combinations} parameter combinations.")
    
    
    if verbose:
        started_at = datetime.datetime.now().replace(microsecond=0)
        print ("Optimization started at: ", end='', flush=True)
        print (started_at.strftime("%Y-%m-%d %H:%M:%S"))

    # Initialize important variables.
    c_index_mean = []
    c_index_std = []
    
    # Loop through all possible parameter combinations.
    for ii, params_curr in enumerate(params_list):

        if verbose:
            print(f"Parameter set {ii+1}/{n_combinations}...")
            print(params_curr)

        # Create RepatedKFold object.        
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

        # To store results.
        c_index_param = []
        
        # Loop through different data partitions.
        for jj, (train_index, val_index) in enumerate(rkf.split(X, Y)):
            
            if verbose:
                print(f"\tIteration {jj+1}/{n_splits*n_repeats}...", end='', flush=True)
                
            # Perform data partition.
            X_train, X_val = X.iloc[train_index,:], X.iloc[val_index,:]
            Y_train, Y_val = Y.iloc[train_index,:], Y.iloc[val_index,:]
            E_train, E_val = E.iloc[train_index,:], E.iloc[val_index,:]
    
            # Create DSK model (with current loop's parameters)
            dsk = deepsurvk.DeepSurvK(n_features=n_features, E=E_train, **params_curr)
            loss = deepsurvk.negative_log_likelihood(E_train)
            dsk.compile(loss=loss)
            callbacks = deepsurvk.common_callbacks()

            # Fit model.
            n_patients_train = X_train.shape[0]
            dsk.fit(X_train, Y_train, 
                    batch_size=n_patients_train,
                    epochs=epochs, 
                    callbacks=callbacks,
                    shuffle=False)
        
            # Generate predictions.
            Y_pred_val = np.exp(-dsk.predict(X_val))
            
            # Compute quality metric (c-index)
            c = deepsurvk.concordance_index(Y_val, Y_pred_val, E_val)
            c_index_param.append(c)
            
            if verbose:
                print(f"\tc-index = {c}")

        # Calculate c-index mean and STD for current parameter set.
        c_index_mean.append(np.nanmean(c_index_param))
        c_index_std.append(np.nanstd(c_index_param))
        
        
    if verbose:
        ended_at = datetime.datetime.now().replace(microsecond=0)
        print ("Optimization ended at: ", end='', flush=True)
        print (ended_at.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"Optimization took {ended_at-started_at}")
        
    
    # Find parameter combination with highest c-index.
    c_index_mean_max = max(c_index_mean)
    idx = c_index_mean.index(c_index_mean_max)

    best_params = params_list[idx]
    
    return best_params
