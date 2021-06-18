# -*- coding: utf-8 -*-
"""Top-level package for DeepSurvK."""

import deepsurvk.network.deepsurvk

from deepsurvk.network.deepsurvk import DeepSurvK, negative_log_likelihood, common_callbacks
from deepsurvk.network.optimization import optimize_hp
from deepsurvk.network.parameters import get_param_list, get_value_from_distribution
from deepsurvk.applications.recommender import recommender_function, get_recs_antirecs_index
from deepsurvk.utils.concordance import concordance_index
from deepsurvk.visualization.dsk_metrics import plot_loss
from deepsurvk.visualization.survival import plot_km_recs_antirecs

from deepsurvk.version import __version__

__version__ = '0.2.0'


__all__ = ['__version__',
           'DeepSurvK',
           'negative_log_likelihood',
           'common_callbacks',
           'optimize_hp',
           'get_param_list',
           'get_value_from_distribution',
           'recommender_function',
           'get_recs_antirecs_index',
           'concordance_index',
           'plot_loss',
           'plot_km_recs_antirecs']
