# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
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
# # 00 - Understanding DeepSurv (using Keras)
# Before anything else, it makes sense to spend some time in understanding
# how the original DeepSurv works. In this notebook we take an example dataset
# and go step by step through the algorithm. Please note that the code 
# here was written with clarity over performance in mind.
#
# ## Preliminaries
#
# Import packages

# %%
import pathlib
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, ActivityRegularization
from tensorflow.keras.optimizers import SGD, Nadam, RMSprop
from tensorflow.keras.regularizers import l2

from lifelines import utils

from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt

import h5py

# import logzero
# from logzero import logger


# %%
# Setup logger.
# logzero.logfile("./logfile.log", maxBytes=1e6, backupCount=2)


# %% [markdown]
# Define paths.

# %%
example_file = '00_understanding_deepsurv'
PATH_DATA = pathlib.Path(r'../deepsurvk/datasets/data')
PATH_MODELS = pathlib.Path(f'./models/')

# Make sure data directory exists.
if not PATH_DATA.exists():
    raise ValueError(f"The directory {PATH_DATA} does not exist.")

# If models directory does not exist, create it.
if not PATH_MODELS.exists():
    PATH_MODELS.mkdir(parents=True)


# %% [markdown]
# ## Get data
# In this case, we will use the Worcester Heart Attack Study (WHAS) dataset.
# For a more detailed description about it, please see the corresponding
# [README](../data/README.md).

# %%
path_data_file = PATH_DATA/'whas.h5'

# Read training data.
with h5py.File(path_data_file, 'r') as f:
    X_train = f['train']['x'][()]
    E_train = f['train']['e'][()]
    Y_train = f['train']['t'][()].reshape(-1, 1)


# Read testing data.
with h5py.File(path_data_file, 'r') as f:
    X_test = f['test']['x'][()]
    E_test = f['test']['e'][()]
    Y_test = f['test']['t'][()].reshape(-1, 1)

# Calculate important parameters.
n_patients_train = X_train.shape[0]
n_features = X_train.shape[1]


# %% [markdown]
# ## Pre-process data
# * Standardization <br>
# First, we need to standardize the input (p. 3).
# Notice how we only use training data for the standardization.
# This done to avoid leakage (using information from
# the testing partition for the model training.)

# %%
X_scaler = StandardScaler().fit(X_train)
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

Y_scaler = StandardScaler().fit(Y_train.reshape(-1, 1))
Y_train = Y_scaler.transform(Y_train)
Y_test = Y_scaler.transform(Y_test)

Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

# %% [markdown]
# * Sorting <br>
# This is important, since we are performing a ranking task.

# %%
sort_idx = np.argsort(Y_train)[::-1]
X_train = X_train[sort_idx]
Y_train = Y_train[sort_idx]
E_train = E_train[sort_idx]


# %% [markdown]
# ## Define the loss function
# DeepSurv's loss function is the average negative log partial likelihood with
# regularization (Eq. 4, p. 3):
#    
# $$l_{\theta} = -\frac{1}{N_{E=1}} \sum_{i:E_i=1} \left( \hat{h}_\theta(x_i) -\log \sum_{j \in {\rm I\!R}(T_i)} \exp^{\hat{h}_\theta(x_j)} \right) + \lambda \cdot \Vert \theta \Vert_2^2 $$
#
# We can see that our loss function depends on three parameters:
# `y_true`, `y_pred`, *and* `E`. Unfortunately, custom loss functions in Keras
# [need to have their signature (i.e., prototype) as](https://keras.io/api/losses/#creating-custom-losses)
# `loss_fn(y_true, y_pred)`. To overcome this, we will use a [small trick](https://github.com/keras-team/keras/issues/2121)
# that is actually well known in the community. This way, we can define the 
# negative log likelihood function as

# %%
def negative_log_likelihood(E):
    def loss(y_true, y_pred):
        
        hazard_ratio = tf.math.exp(y_pred)        
        log_risk = tf.math.log(tf.math.cumsum(hazard_ratio))
        uncensored_likelihood = tf.transpose(y_pred) - log_risk
        censored_likelihood = uncensored_likelihood * E
        neg_likelihood_ = -tf.math.reduce_sum(censored_likelihood)

        # TODO
        # For some reason, adding num_observed_events does not work.
        # Therefore, for now we will use it as a simple factor of 1.
        # Is it really needed? Isn't it just a scaling factor?
        # num_observed_events = tf.math.cumsum(E)
        # num_observed_events = tf.cast(num_observed_events, dtype=tf.float32)
        num_observed_events = tf.constant(1, dtype=tf.float32)
        
        neg_likelihood = neg_likelihood_ / num_observed_events        
        
        return neg_likelihood
    
    return loss


# %% [markdown]
# with regularization added further on (as part of the network architecture).
#
# ## Define model parameters
# Nothing spectacular here. You can see these are pretty standard parameters.
# We will use the values reported in Table 2 (p. 10).
#
# If you decide to try a different dataset, be sure to change these
# accordingly!

# %%
activation = 'relu'
n_nodes = 48
learning_rate = 0.067
l2_reg = 16.094
dropout = 0.147
lr_decay =  6.494e-4
momentum = 0.863


# %% [markdown]
#
# ## Model construction
# Now we can build the model. We will do this using the `Sequential` 
# constructor, thus adding layer by layer.
#
# The initialization of the nodes weights can be done in many different
# ways. In the original DeepSurv implementation, they used [Glorot
# with weights sampled from the uniform distribution](https://github.com/jaredleekatzman/DeepSurv/blob/198bb2375ea2d2cad93e568ffc550889366494ef/deepsurv/deep_surv.py#L78),
# as proposed by [Glorot and Bengio (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).
# Therefore, we will stick with that initialization as well.
#
# Notice that this architecture works for the sample dataset (WHAS).
# It is slightly different for each dataset (mainly the optimizer and 
# number of hidden layers).

# %%
# Create model
model = Sequential()
model.add(Dense(units=n_features, activation=activation, kernel_initializer='glorot_uniform', input_shape=(n_features,)))
model.add(Dropout(dropout))
model.add(Dense(units=n_nodes, activation=activation, kernel_initializer='glorot_uniform'))
model.add(Dropout(dropout))
model.add(Dense(units=n_nodes, activation=activation, kernel_initializer='glorot_uniform'))
model.add(Dropout(dropout))
model.add(Dense(units=1, activation='linear', kernel_initializer='glorot_uniform', kernel_regularizer=l2(l2_reg)))
model.add(ActivityRegularization(l2=l2_reg))

# Define the optimizer
# Nadam is Adam + Nesterov momentum
# optimizer = Nadam(learning_rate=learning_rate, decay=lr_decay, clipnorm=1) 
optimizer = Nadam(learning_rate=learning_rate, decay=lr_decay)

# Compile the model and show a summary of it
model.compile(loss=negative_log_likelihood(E_train), optimizer=optimizer)
model.summary()


# %% [markdown]
# Sometimes, the computation of the loss yields a `NaN`, which makes the whole
# output be `NaN` as well. I haven't identified a pattern, actually I think
# it is quite random. This could be due to a variety of reasons, including
# model parametrization (however, I don't really want to use different 
# parameters than those reported), maybe even unfortunate parameter 
# initialization. Therefore, we will use a technique called "Early Stopping".
#
# In this case, we will train the model until the number of epochs is reached
# *or* until the loss is an `NaN`. After that, training is stopped. Then,
# we will selected and use the model that yielded the smallest lost.
#
# We can achieve this very easily using [callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback)

# %%
callbacks = [tf.keras.callbacks.TerminateOnNaN(),
             tf.keras.callbacks.ModelCheckpoint(str(PATH_MODELS/f'{example_file}.h5'), monitor='loss', save_best_only=True, mode='min')]

# %% [markdown]
# ## Model fitting
# Now we can fit the DeepSurv model! Notice how we use the whole set of 
# patients in a batch. Furthermore, be sure that `shuffle` is set to `False`, 
# since order is important in predicting ranked survival.

# %%
epochs = 500
history = model.fit(X_train, Y_train, 
                    batch_size=n_patients_train, 
                    epochs=epochs, 
                    callbacks=callbacks,
                    shuffle=False)

# %% [markdown]
# We can see how the loss changed with the number of epochs.

# %%
fig, ax = plt.subplots(1, 1, figsize=[5, 5])
plt.plot(history.history['loss'], label='train')
ax.set_xlabel("No. epochs")
ax.set_ylabel("Loss [u.a.]")


# %% [markdown]
# During training, we saved the model with the lowest loss value (i.e., Early Stop).
# Now, we need to load it. Since we defined our own custom function,
# it is important to [use the `compile=False` parameter](https://github.com/keras-team/keras/issues/5916#issuecomment-592269254).

# %%
model = load_model(PATH_MODELS/f'{example_file}.h5', compile=False)

# %% [markdown]
# ## Model predictions
# Finally, we can generate predictions using the DeepSurv model.

# %%
Y_pred_train = np.exp(-model.predict(X_train))
c_index_train = utils.concordance_index(Y_train, Y_pred_train, E_train)
print(f"c-index of training dataset = {c_index_train}")

Y_pred_test = np.exp(-model.predict(X_test))
c_index_test = utils.concordance_index(Y_test, Y_pred_test, E_test)
print(f"c-index of testing dataset = {c_index_test}")

# %% [markdown]
# We can see that these numbers are within the ballpark estimate of what is
# reported in the original paper for this dataset (0.86-0.87, Table 1, p. 6).
