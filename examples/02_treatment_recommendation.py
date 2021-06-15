# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# Before we begin, we will change a few settings to make the notebook look a bit prettier

# %% language="html"
# <style> body {font-family: "Calibri", cursive, sans-serif;} </style>


# %% [markdown]
#
# # 02 - Treatment Recommendation
# One of the original DeepSurv's coolest features is that it can be used as a 
# personalized treatment recommender. In this notebook, we will see how this 
# works in DeepSurvK.
#
# This notebook assumes that you have gone through the [basics of DeepSurv](./00_understanding_deepsurv.ipynb)
# as well as [DeepSurvK's basic usage](./01_deepsurvk_quickstart.ipynb)
#
# ## Preliminaries
#
# Import packages

# %%
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import deepsurvk
from deepsurvk.datasets import load_rgbsg


# %% [markdown]
# ## Fit model
# The first step is to generate and fit a DeepSurvK model.
# We will do so in the same manner as we did before.
#
# ### Get data
# We will use the RGBSG dataset, since this is the one that was used as an
# example in the original paper (p. 8)

# %%
X_train, Y_train, E_train = load_rgbsg(partition='training')
X_test, Y_test, E_test = load_rgbsg(partition='testing')

# Calculate important parameters.
n_patients_train = X_train.shape[0]
n_features = X_train.shape[1]

# %% [markdown]
# ### Pre-process data

# %%
# Standardization
cols_standardize = ['grade', 'age', 'n_positive_nodes', 'progesterone', 'estrogen']
X_ct = ColumnTransformer([('standardizer', StandardScaler(), cols_standardize)])
X_ct.fit(X_train[cols_standardize])

X_train[cols_standardize] = X_ct.transform(X_train[cols_standardize])
X_test[cols_standardize] = X_ct.transform(X_test[cols_standardize])

Y_scaler = StandardScaler().fit(Y_train)
Y_train['T'] = Y_scaler.transform(Y_train)
Y_test['T'] = Y_scaler.transform(Y_test)


# %%
# Sorting
sort_idx = np.argsort(Y_train.to_numpy(), axis=None)[::-1]
X_train = X_train.loc[sort_idx, :]
Y_train = Y_train.loc[sort_idx, :]
E_train = E_train.loc[sort_idx, :]


# %% [markdown]
# ### DeepSurvK modelling
# We will use the parameters that correspond to the RGBSG dataset,
# as reported in Table 1.

# %%
params = {'n_layers':1,
          'n_nodes':8,
          'activation':'selu',
          'learning_rate':0.154,
          'decays':5.667e-3,
          'momentum':0.887,
          'l2_reg':6.551,
          'dropout':0.661,
          'optimizer':'nadam'}

# %%
dsk = deepsurvk.DeepSurvK(n_features=n_features, 
                          E=E_train,
                          **params)

# %%
loss = deepsurvk.negative_log_likelihood(E_train)
dsk.compile(loss=loss)

# %%
callbacks = deepsurvk.common_callbacks()

epochs = 1000
history = dsk.fit(X_train, Y_train, 
                  batch_size=n_patients_train,
                  epochs=epochs, 
                  callbacks=callbacks,
                  shuffle=False)

# %%
deepsurvk.plot_loss(history)

# %%
# Perform predictions for test data (sanity check)
Y_pred_test = np.exp(-dsk.predict(X_test))
c_index_test = deepsurvk.concordance_index(Y_test, Y_pred_test, E_test)
print(f"c-index of testing dataset = {c_index_test}")


# %% [markdown]
# ## Treatment recommendation
# The original paper has a very clear explanation of how the treatment
# recommender system works, which is as follows:
#
# > We assume each treatment $i$ to have an independent risk function 
# $e^{h_i(x)}$. [...] For any patient, the [model] should be able to 
# accurately predict the log-risk $h_i(x)$ of being prescribed a given
# treatment $i$. Then, based on the assumption that each individual
# has the same baseline hazard function $\lambda_0(t)$, we
# can take the log of the hazards ratio to calculate the personal
# risk-ratio of prescribing one treatment option over
# another. We define this difference of log hazards as the
# recommender function $rec_{ij}(x)$:
#
# $rec_{ij}(x) = \log \left( \frac{\lambda_0(t) e^{h_i(x)}}{\lambda_0(t) e^{h_j(x)}} \right) $
#
# $rec_{ij}(x) = h_i(x) - h_j(x) $
#
# DeepSurvK provides the function `recommender_function`, which 
# allows calculating $rec_{ij}(x)$ in a very easy way:

# %%
rec_ij = deepsurvk.recommender_function(dsk, X_test, 'horm_treatment')

# %% [markdown]
# > The recommender function can be used to provide personalized treatment 
# recommendations. We first pass a patient through the network once in 
# treatment group $i$ and again in treatment group $j$ and take the 
# difference. When a patient receives a positive recommendation $rec_{ij}(x)$,
# treatment $i$ leads to a higher risk of death than treatment $j$. Hence, 
# the patient should be prescribed treatment $j$. Conversely, a negative 
# recommendation indicates that treatment $i$ is more effective and leads to 
# a lower risk of death than treatment $j$, and we recommend treatment $i$.
#
# DeepSurv also has a function to find these subsets of patients 
# (recommendation and anti-recommendation):

# %%
recommendation_idx, _ = deepsurvk.get_recs_antirecs_index(rec_ij, X_test, 'horm_treatment')

# %% [markdown]
# `get_recs_antirecs_index` gives as a second output `antirecommendation_idx`.
# However, it is nothing else than the negated version of 
# `recommendation_idx`. Therefore, we will ignore the former and stick with
# the later.
#
# Finally, we can generate Kaplan-Meier (KM) curves for each patient group.
# To do so, first we need to invert the transformation we did previously
# on `Y_test` (to bring it back to proper time units).

# %%
Y_test_original = Y_test.copy(deep=True)
Y_test_original['T'] = Y_scaler.inverse_transform(Y_test)

# %% [markdown]
# DeepSurvK provides a function to quickly generate such plot.
# Notice how this visualization pretty much matches Fig. 6a of
# the original paper.

# %%
deepsurvk.plot_km_recs_antirecs(Y_test_original, E_test, recommendation_idx)

# %% [markdown]
# We can see that the KM curve of the patients that were treated according
# to the model's recommendation is higher than that of patients that were
# *not*. This is confirmed by the log-rank statistic with a $p$-value
# of 0.0034.
