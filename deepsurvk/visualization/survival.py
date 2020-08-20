# -*- coding: utf-8 -*-
"""
survival.py
Different functions to generate survival-related visualizations.
"""
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import lifelines

__all__ = ['plot_km_recs_antirecs']

#%%
# A few tweeks to make plots pretty.

sns.set(style="whitegrid")
sns.set(font_scale=1.75)
sns.set_style('ticks')
mpl.rcParams['font.sans-serif'] = "Calibri"
mpl.rcParams['font.family'] = "sans-serif"
plt.rc('axes.spines', top=False, right=False)


#%%
def plot_km_recs_antirecs(T, E, recommendation_idx, fig=None, ax=None, xlim=None, ylim=None, show_risk=False):
    """
    Plot KM curves for (anti)recommendation patients.
    
    Parameters
    ----------
    T: pandas DataFrame
        It needs to have column 'T'
    E: pandas DataFrame
        It needs to have column 'E'
    recommendation_idx: boolean array
        Array as given by get_recs_antirecs_index. It is True for
        recommendation patients. 
    fig: figure handle (optional)
    ax: axes handle (optional)
    xlim: list (two elements, optional)
        x-axis boundaries.
    ylim: list (two elements, optional)
        y-axis boundaries. If left as None, defaults to [0, 1]
    show_risk: boolean (optional)
        Indicate if the number of patients at risk should be included below
        the axis (True) or not (False, default).
            
    Returns
    -------
    tuple
        The first element corresponds to the figure handle.
        The second element correpsonds to the axes handle.
    """
    
    # Create figure (if necessary).
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1, figsize=[12, 6])
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()

    # Initialize variables.
    kmf_list = []
    T_list = []
    C_list = []
    
    # For each label, apply KMF and plot.        
    labels = ['recommendation', 'anti-recommendation']
    
    for label in labels:

        # Perform proper selection.
        if label=='recommendation':
            T_curr = T.loc[recommendation_idx, :]
            E_curr = E.loc[recommendation_idx, :]
        elif label=='anti-recommendation':
            T_curr = T.loc[~recommendation_idx, :]
            E_curr = E.loc[~recommendation_idx, :]

        # Create Kaplan Meier Fitter and fit.
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(T_curr, E_curr, label=label.capitalize())
        
        # Plot KM curve.
        ax = kmf.plot(ax=ax, linewidth=5, legend=True)
        ax.legend(loc='best', frameon=False, fontsize='small')

        kmf_list.append(kmf)
        T_list.append(T_curr)
        C_list.append(E_curr)

    
    # Perform statistical analysis (log-rank test).
    results = lifelines.statistics.logrank_test(T_list[0], T_list[1], C_list[0], C_list[1], alpha=0.95)
    results.print_summary(style='ascii', decimals=4)

    # Calculate p-value text position and display.
    if ylim==None:
        y_pos = 0.1
    else:
        y_pos = 0.1 + min(ylim) + ((max(ylim) - min(ylim))*0.1)
        
    if results.p_value < 0.001:
        p_value_text = "$p$ < 0.001"
    else:
        p_value_text = f"$p$ = {results.p_value:.4f}"
    ax.text(T['T'].min()*10, y_pos, p_value_text, fontsize='small')
    
    # Format x-axis ticks here.
    # xticks = np.arange(T['T'].min(), T['T'].max())
    # xticks_float = xticks
    # xticks_floor = np.floor(xticks_float)
    # xticks_ceil = np.ceil(xticks_float)
    # xticks = np.unique(np.concatenate([xticks_floor, xticks_ceil], axis=None))
    # # Remove unnecesary ticks.
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticks.astype(int))
    if xlim!=None:
        ax.set_xlim(np.array(xlim))    
    if ylim!=None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([0, 1])
    ax.set_ylabel("Survival probability", weight='bold')
    
    # Add risk counts.
    if show_risk:
        lifelines.plotting.add_at_risk_counts(kmf_list[0], kmf_list[1], ax=ax)
        
    # X-axis label is set here to be sure it is show correctly even if
    # patients at risk will be shown.
    ax.set_xlabel("Time", weight='bold')
        
    return fig, ax