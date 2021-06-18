
# -*- coding: utf-8 -*-
"""
parameters.py
Functions for manipulating parameters.
"""
import numpy as np
import scipy as sp
import random
import itertools

__all__ = ['get_param_list',
           'get_value_from_distribution']

#%%
def get_param_list(params, mode='grid', n_iter=25):
    """
    Get a list with all the parameter combinations that will be tested
    for optimization.
    
    Parameters
    ----------
    params: dictionary
        Each key corresponds to a parameter. 
        The values correspond to a list of parameters to be explored.
        In the case of 'grid', all possible parameter combinations will
        be explored.
        In the case of 'random', a determined number random distribution of parameters
        
    mode: string
        Possible values are:
            'grid' (default)
            'random'
            
    n_iter: int (optional)
        Number of parameter setitngs that will be sampled. 
        Only valid for 'random' mode. Otherwise, ignored.
        Default value is 25.
        Notice there is a trade off between runtime and quality of the 
        solution. 
        
    Returns
    -------
    param_list: list
        List of dictionaries. Each dictionary has a parameter combination 
        to try.
    """
    
    # Generating a list of dictionaries with all parameter combinations to try.
    if mode == 'grid':
        # In this case, we generate a list of dictionaries of ALL
        # possible parameter value combinations.
        # Trick from https://stackoverflow.com/a/61335465/948768
        keys, values = zip(*params.items())
        param_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
    elif mode == 'random':
        # In this case, we generate a list of dictionaries with random
        # combinations of parameter values.
        param_list = []
        for hh in range(n_iter):
            
            # Initialization.
            param_dict = {}
            
            for key, value in zip(params.keys(), params.values()):
                param_dict[key] = get_value_from_distribution(key, value)
                    
            # Append the generated dictionary to the list.
            param_list.append(param_dict)        

    else:
        raise ValueError("Invalid parameter optimization mode. Possible values are 'grid' and 'random'.")
        
    return param_list


#%%
def get_value_from_distribution(param, param_values):
    """
    Get a random value for a parameter.
    
    Parameters
    ----------
    param: string
        Name of the parameter
        
    param_values: list
        List with parameter boundaries/possible values.
        For numeric parameters, the smallest value will be
        the lower boundary, while the largest one will be the higher one.
        Any other value will be ignored.
        For non-numeric parameters, the value will be drawn at
        random from the given values.
        In all cases, if only one value is given, 
        the same value will be returned, since that means that the user
        wanted that parameter to be fixed.
            
    Returns
    -------
    value: numeric or string (depending on the parameter)
        Random value drawn from a distribution for the given parameter.
        The type of value will depend on the parameter:
            n_layers        Integer drawn from a uniform distribution
            n_nodes         Integer drawn from a uniform distribution
            activation
            learning_rate
            decay
            momentum
            l2_reg
            dropout
            optimizer
    """
    if len(param_values) == 0:
        # Check for empty lists.
        raise ValueError("Parameter list must have at least one element.")
        
    elif len(param_values) == 1:
        # If a parameter was just given one single value to try,
        # that means that the user wants to fix it. Thus, it won't
        # be considered for randomization.
        value = param_values[0]
        
    else:
        # In the cases where more than one value was given,
        # we will obtain a random value. 
        
        # There parameters are integers.
        if (param == 'n_layers') or (param == 'n_nodes'):
            
            # In this case, make sure that values are round numbers.
            param_values = [round(x) for x in param_values]
            
            # Get boundaries.
            low_boundary, high_boundary = _get_numeric_boundaries(param_values)
            
            # Get values from distribution.
            rng = np.random.default_rng()
            value = rng.integers(low_boundary, high_boundary, endpoint=True)
            
        if param == 'learning_rate':
            # Get boundaries.
            low_boundary, high_boundary = _get_numeric_boundaries(param_values)
            
            # Validate values of boundaries.
            if low_boundary < 10**-6:
                print("WARNING: lower boundary of learning_rate is too small (<10^-6)")
            if high_boundary > 1:
                print("WARNING: high boundary of learning_rate is > 1. It will be set to 1.")
                high_boundary = 1
                
            # In this case, generate a logarithmically uniform spaced 
            # set of numbers and sample from there.
            param_values_ = sp.stats.loguniform.rvs(low_boundary, high_boundary, size=1000)
            value = random.choice(param_values_)
            
        # There parameters should be between 0 and 1.
        if (param == 'momentum') or (param == 'dropout'):
            # Get boundaries.
            low_boundary, high_boundary = _get_numeric_boundaries(param_values)
            
            # Validate values of boundaries.
            if low_boundary < 0:
                print("WARNING: lower boundary of " + param + " is too small (< 0). Will be set to 0.")
                low_boundary = 0
            if high_boundary > 1:
                print("WARNING: high boundary of " + param + " is > 1. It will be set to 1.")
                high_boundary = 1
                
            # In this case, generate a linearly uniform spaced 
            # set of floats and sample from there.
            rng = np.random.default_rng()
            value = rng.uniform(low_boundary, high_boundary)
            
        # There parameters should be larger than 0.
        if (param == 'l2_reg') or (param == 'decay'):
            # Get boundaries.
            low_boundary, high_boundary = _get_numeric_boundaries(param_values)
            
            # Validate values of boundaries.
            if low_boundary < 0:
                print("WARNING: lower boundary of " + param + " is negative. Will be set to 0.")
                low_boundary = 0
                
            # In this case, generate a linearly uniform spaced 
            # set of floats and sample from there.
            rng = np.random.default_rng()
            value = rng.uniform(low_boundary, high_boundary)
                
        # These parameters are non-numeric.
        elif (param == 'activation') or (param == 'optimizer'):
            value = random.choice(param_values)
    
    return value


#%%
def _get_numeric_boundaries(boundaries):
    low_boundary = min(boundaries)
    high_boundary = max(boundaries)

    return low_boundary, high_boundary
    