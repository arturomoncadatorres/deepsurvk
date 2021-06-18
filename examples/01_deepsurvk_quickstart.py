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
# # 01 - DeepSurvK Quickstart
# In this notebook, I will show DeepSurvK's basic functionality.
#
# Before going forward, I recommend you check the previous notebook,
# ["Understanding DeepSurv"](./00_understanding_deepsurv.ipynb). 
# There, you will learn the working principles of the original DeepSurv 
# algorithm. Furthermore, there are also some useful usage recommendations 
# that also apply here. However, I will just mention them without going
# into the details.
#
# ## Preliminaries
#
# Import packages

# %%
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import deepsurvk
from deepsurvk.datasets import load_whas


# %% [markdown]
# ## Get data
# For convenience, DeepSurvK comes with DeepSurv's original datasets. 
# This way, we can load sample data very easily (notice the import at the
# top).

# %%
X_train, Y_train, E_train, = load_whas(partition='training', data_type='np')
X_test, Y_test, E_test = load_whas(partition='testing', data_type='np')

# %% [markdown]
# These `training` and `testing` partitions correspond to the original
# partitions used in DeepSurv's paper. 
# However, you could also load the complete dataset using
# `partition='complete'` and partition it as you wish (e.g., using
# sklearn's [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html))

# %%
# Calculate important parameters.
n_patients_train = X_train.shape[0]
n_features = X_train.shape[1]


# %% [markdown]
# ## Pre-process data
# Data pre-processing is an important step. However, DeepSurvK leaves this
# to the user, since it depends very much on the data themselves.
# As mentioned in the [previous notebook]((./00_understanding_deepsurv.ipynb)), 
# at the very least, I would recommend doing standardization and sorting:

# %%
# Standardization
X_scaler = StandardScaler().fit(X_train)
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

Y_scaler = StandardScaler().fit(Y_train.reshape(-1, 1))
Y_train = Y_scaler.transform(Y_train)
Y_test = Y_scaler.transform(Y_test)


# %%
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()


# %% [markdown]
# > Notice that if you read/have your data as a `pandas` DataFrame, you will
# > get an error when reshaping `Y_train` (see [issue #81](https://github.com/arturomoncadatorres/deepsurvk/issues/81)). 
# > That is because a DataFrame doesn't have the `reshape` attribute.
# >
# > In such case, you need to do the reshaping as follows:
# >
# > ```
# > Y_scaler = StandardScaler().fit(Y_train.values.reshape(-1, 1))
# > ```

# %%
# Sorting
sort_idx = np.argsort(Y_train)[::-1]
X_train = X_train[sort_idx]
Y_train = Y_train[sort_idx]
E_train = E_train[sort_idx]

# %% [markdown]
# > Notice that if you read/have your data as a `pandas` DataFrame, you will
# > get an error when sorting (see [issue #82](https://github.com/arturomoncadatorres/deepsurvk/issues/82)). 
# > That is because a DataFrame cannot be sorted like this.
# >
# > In such case, you need to do the sorting as follows:
# >
# > ```
# > X_train = X_train.values[sort_idx]
# > ...
# > ```
#
# ## Create a DeepSurvK model
# When creating an instance of a DeepSurvK model, we can also define its 
# parameters. The only mandatory parameters are `n_features` and `E`.
# If not defined, the rest of the parameters will use a default.
# This is, of course, far from optimal, since (hyper)parameter tuning
# has a *huge* impact on model performance. However, we will deal
# with that later.

# %%
dsk = deepsurvk.DeepSurvK(n_features=n_features, E=E_train)

# %% [markdown]
# Since DeepSurvK is just a Keras model, we can take advantage of all the
# perks and tools that come with it. For example, we can get an overview
# of the model architecture very easily.

# %%
dsk.summary()

# %% [markdown]
# ## Callbacks
# As mentioned earlier, it is practical to use Early Stopping in the
# case of NaNs in loss values. Additionally, it is also a good idea
# to use the model that during the training phase yields the lowest loss
# (which isn't necessarily the one at the end of the training)
#
# Both of these practices can be achieved using [callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback).
# DeepSurvK provides a method to generate these two specific callbacks.

# %%
callbacks = deepsurvk.common_callbacks()
print(callbacks)

# %% [markdown]
# Needless to say that you can define your own callbacks as well, of course.
#
# ## Model fitting
# After this, we are ready to actually fit our model (as any Keras model).

# %%
epochs = 1000
history = dsk.fit(X_train, Y_train, 
                  batch_size=n_patients_train,
                  epochs=epochs, 
                  callbacks=callbacks,
                  shuffle=False)


# %% [markdown]
# > In some cases, it has been reported that while fitting a model,
# > the [loss goes to a `NaN` very early](https://github.com/arturomoncadatorres/deepsurvk/issues/83),
# > making the training process unfeasible, even with the previously defined
# > callback. I haven't been able to replicate that issue consistently.
# > 
# > However, this issue has also [been reported in the original DeepSurv](https://github.com/jaredleekatzman/DeepSurv/issues/14).
# > Apparently, a potentially good solution for this is to *not* 
# > standardize your data during the pre-procesing, but rather
# > normalizing it (i.e., make sure that features are in the range 0-1).
# > However, remember that scaling is particularly sensitive to
# > outliers, so be careful!

# %% [markdown]
# DeepSurvK provides a few wrappers to generate visualizations that are
# often required fast and easy.

# %%
deepsurvk.plot_loss(history)

# %% [markdown]
# ## Model predictions
# Finally, we can generate predictions using our model.
# We can evaluate them using the c-index.

# %%
Y_pred_train = np.exp(-dsk.predict(X_train))
c_index_train = deepsurvk.concordance_index(Y_train, Y_pred_train, E_train)
print(f"c-index of training dataset = {c_index_train}")

Y_pred_test = np.exp(-dsk.predict(X_test))
c_index_test = deepsurvk.concordance_index(Y_test, Y_pred_test, E_test)
print(f"c-index of testing dataset = {c_index_test}")
