# -*- coding: utf-8 -*-
"""Top-level package for DeepSurvK."""

import deepsurvk.network.deepsurvk

from deepsurvk.network.deepsurvk import DeepSurvK, DeepSurvK_kt, negative_log_likelihood, common_callbacks
from deepsurvk.applications.recommender import recommender_function, get_recs_antirecs_index
from deepsurvk.utils.concordance import concordance_index, concordance_index_objective
from deepsurvk.visualization.dsk_metrics import plot_loss
from deepsurvk.visualization.survival import plot_km_recs_antirecs

from deepsurvk.version import __version__

__version__ = '0.1.2'


__all__ = ['__version__',
           'DeepSurvK',
           'DeepSurvK_kt',
           'negative_log_likelihood',
           'common_callbacks',
           'recommender_function',
           'get_recs_antirecs_index',
           'concordance_index',
           'concordance_index_objective',
           'plot_loss',
           'plot_km_recs_antirecs']
