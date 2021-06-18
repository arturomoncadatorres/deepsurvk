# -*- coding: utf-8 -*-
"""
concordance.py
Different functions (and wrappers) related to concordance metrics.
"""
import numpy as np
from lifelines.utils import concordance_index as c_index

__all__ = ['concordance_index']

#%%
def concordance_index(y_true, y_pred, E):
    """
    Computes the concordance index (aka c-index).
    This is a wrapper of the concordance_index function in lifelines.
    This is done just for convenience, saving the user an additional
    import of lifelines [1].
    
    Parameters
    ----------
    y_true: 
        
    y_pred: 
        
    E:

            
    Returns
    -------
    c: float
        The c-index
        
    References
    ----------
    [1] https://github.com/CamDavidsonPilon/lifelines
    """
    # Check for NaNs.
    if np.isnan(np.sum(y_pred)):
        c = np.nan
    else:
        c = c_index(y_true, y_pred, E)
    return c
