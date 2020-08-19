# -*- coding: utf-8 -*-
"""
dsk_metrics.py
Different functions to plot DeepSurvK metrics.
"""
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

__all__ = ['plot_loss']


#%%
# A few tweeks to make plots pretty.

sns.set(style="whitegrid")
sns.set(font_scale=1.25)
sns.set_style('ticks')
mpl.rcParams['font.sans-serif'] = "Calibri"
mpl.rcParams['font.family'] = "sans-serif"
plt.rc('axes.spines', top=False, right=False)


#%%
def plot_loss(history):
    """
    Plot loss progression.
    
    Parameters
    ----------
    history: a Keras history object
        History object generated using model.fit (i.e., training) [1]
            
    Returns
    -------
    fig_ax: tuple
        The first element corresponds to the figure handle.
        The second element correpsonds to the axes handle.
        
    References
    ----------
    [1] https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    """
    fig, ax = plt.subplots(1, 1, figsize=[5, 5])
    plt.plot(history.history['loss'], label='train')
    ax.set_xlabel("No. epochs")
    ax.set_ylabel("Loss [u.a.]")

    return fig, ax