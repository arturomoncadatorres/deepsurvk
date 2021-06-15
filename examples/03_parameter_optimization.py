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
# # 03 - Parameter Optimization
# So far, we have used parameters that have been previously reported (namely
# in the original paper). However, more likely than not, you will be using
# data of your own, which will require tuning the model's (hyper)parameters.
#
# As we all know, hyperparameter tuning can be almost an art of itself. 
# However, fortunately TensorFlow 2.0 has a hyperparameter tuner in
# form of [Keras Tuner](https://keras-team.github.io/keras-tuner/),
# In this notebook, we will see how this works in DeepSurvK.
#
# This notebook assumes that you have gone through the [basics of DeepSurv](./00_understanding_deepsurv.ipynb)
# as well as [DeepSurvK's basic usage](./01_deepsurvk_quickstart.ipynb)
#
# ## Preliminaries
#
# Import packages

# %%
import pathlib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import deepsurvk
from deepsurvk.datasets import load_rgbsg

# %% [markdown]
# Define paths.

# %%
PATH_MODELS = pathlib.Path(f'./models/')

# If models directory does not exist, create it.
if not PATH_MODELS.exists():
    PATH_MODELS.mkdir(parents=True)


# %% [markdown]
# ## Get data
# We will use the RGBSG dataset.

# %%
X_train, Y_train, E_train = load_rgbsg(partition='training')
X_test, Y_test, E_test = load_rgbsg(partition='testing')

# Calculate important parameters.
n_patients_train = X_train.shape[0]
n_features = X_train.shape[1]

# %% [markdown]
# ## Pre-process data

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


# %% [markdown]
# ## (Hyper)parameter optimization
#
# We will define the parameters that we wish to explore as a dictionary.
# Notice that all parameters must be given in a list, even if they consist
# of a single value.
#
# We can define the number of epochs (`epochs`) in that same dictionary as well.
# Technically, we won't optimize this parameter. However, we might
# want to set it to a given value. In this case, provide only a single value.
# If more are given, only the first will be considered. If it isn't 
# defined, a default value of 1000 is used.
#
# For this example, we will fix most of the reported parameters 
# for the RGBSG dataset and optimize only three of them: 
# - `n_layers` - 1, 4
# - `n_nodes` - 2, 8
# - `activation` - `relu`, `selu`

# %%
params = {'epochs':[500],
          'n_layers':[1, 4],
          'n_nodes':[2, 8], 
          'activation':['relu', 'selu'],
          'learning_rate':[0.154],
          'decay':[5.667e-3],
          'momentum':[0.887],
          'l2_reg':[6.551],
          'dropout':[0.661],
          'optimizer':['nadam']}

# %% [markdown]
# This will results in testing a very small number of 
# possible combinations (8). Using default values, the optimization 
# will use 3 folds (as reported in the original paper) and 5 repetitions. 
# So for each parameter combination, a model will be fitted and evaluated 
# 15 times. In total, this will result in 120 fits. Notice how this can 
# result in an exponential increase of computational time required 
# depending on the number of parameters to be optimized. 
# Be careful with this, especially if you are using a large dataset!
#
# Then, we will obtain the best parameters using DeepSurvK's `optimize_hp`.
# Currently, a raw version of grid search is implemented. In the future, I 
# plan to expand this to a randomized search.
#
# > *Why not use an existing (hyper)parameter optimization tool?*
# > 
# > (Hyper)parameter optimization is a well known issue of deep learning
# > models. There are a few tools out there that are designed for this
# > purpose, such as [Talos](https://github.com/autonomio/talos),
# > using a [Scikit wrapper](https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn)
# > and even Keras's own [Keras Tuner](https://github.com/keras-team/keras-tuner).
# > Unfortunately, I couldn't manage to get any of these tools working for
# > DeepSurvK. This was mainly because DeepSurvK (since it is a survival task)
# > depends not only of `X` and `Y`, but also `E`. This is a problem
# > since none of these options support this extra parameter.
# > Maybe I'm wrong and this could be fixed in a future version.
# > Contributions are always welcome.
#
# In theory, we should get the reported values for these parameters 
# (`n_layers = 1`, `n_nodes = 8`, `activation = selu`).


# %%
best_params = deepsurvk.optimize_hp(X_train, Y_train, E_train, 
                                    mode='grid', 
                                    n_splits=3, 
                                    n_repeats=5, 
                                    verbose=True, 
                                    **params)

print(best_params)

# %% [markdown]
# This looks good. Now, as usual, we can just create a new model with the
# optimized parameters, fit it, and generate predictions.

# %%
dsk = deepsurvk.DeepSurvK(n_features=n_features, E=E_train, **best_params)
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
Y_pred_test = np.exp(-dsk.predict(X_test))
c_index_test = deepsurvk.concordance_index(Y_test, Y_pred_test, E_test)
print(f"c-index of testing dataset = {c_index_test}")
