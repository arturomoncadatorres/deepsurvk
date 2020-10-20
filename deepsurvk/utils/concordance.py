# -*- coding: utf-8 -*-
"""
concordance.py
Different functions (and wrappers) related to concordance metrics.
"""

from lifelines.utils import concordance_index as c_index

__all__ = ['concordance_index', 'concordance_index_objective']

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
    c = c_index(y_true, y_pred, E)
    return c


#%%
def concordance_index_objective(E):
    """
    This function is equivalent to concordance_index (i.e., computes the 
    c-index using the concordance_index function in lifelines [1]).
    However, it is rewritten in a different shape to be able to use 
    it as an objective function when defining a DeepSurvK_kt model.
    
    Parameters
    ----------
    y_true: 
        
    y_pred: 
        
    E:

            
    Returns
    -------
    _concordance_index:
        
    References
    ----------
    [1] https://github.com/CamDavidsonPilon/lifelines
    """
    def _concordance_index(y_true, y_pred):
        c = c_index(y_true, y_pred, E)
    return _concordance_index