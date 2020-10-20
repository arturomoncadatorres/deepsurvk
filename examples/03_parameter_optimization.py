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
# # 03 - Parameter Optimization
# So far, we have used parameters that have been previously reported (namely
# in the original paper). However, more likely than not, you will be using
# data of your own, which will require tuning the model's hyperparameters.
#
# As we all know, hyperparameter tuning can be almost an art of itself. 
# However, fortunately TensorFlow 2.0 has a hyperparameter tuner in
# form of [Keras Tuner](https://keras-team.github.io/keras-tuner/),
# In this notebook, we will see how this works in DeepSurvK.
#
# This notebook assumes that you have gone through the [basics of DeepSurv](./00_understanding_deepsurv.ipynb)
# as well as [DeepSurvK's basic usage](./00_using_deepsurvk.ipynb)
#
# ## Preliminaries
#
# Import packages

# %%
import pathlib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import kerastuner as kt
from kerastuner import HyperParameters
from kerastuner.tuners import RandomSearch

import deepsurvk
from deepsurvk.datasets import load_rgbsg

# import logzero
# from logzero import logger


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


# %%
# Sorting
sort_idx = np.argsort(Y_train.to_numpy(), axis=None)[::-1]
X_train = X_train.loc[sort_idx, :]
Y_train = Y_train.loc[sort_idx, :]
E_train = E_train.loc[sort_idx, :]


# %% [markdown]
# ## Hyperparameter tuning.
# The important part of this notebook.
#
# First, we need to create a HyperParameters object `hp`.
# Since `n_features` is crucial to the network's architecture, we will 
# "pack" it into `hp`.

# %%
hp = HyperParameters()
hp.Fixed(name='n_features', value=n_features)


# %% [markdown]
# Then, we can 

# %%
dsk_kt = deepsurvk.DeepSurvK_kt(hp)

loss = deepsurvk.negative_log_likelihood(E_train)
dsk_kt.compile(loss=loss)

#%%%
def c_index(y_true, y_pred):
    
    total = 0
    matches = 0
    for ii in range(len(y_true)):
        for jj in range(len(y_true)):
    
            if y_true[jj] > 0 and abs(y_true[ii]) > y_true[jj]:
                total += 1
                if y_pred[jj] > y_pred[ii]:
                    matches += 1
    return matches/total


# %%
loss = deepsurvk.negative_log_likelihood(E_train)
DeepSurvK_kt.compile(loss=loss)


# %%
def DeepSurvK_kt(hp):
    """
    Create a Keras model using the DeepSurv architecture, as originally
    proposed in [1]. This implementation uses (hyper)parameters as optimized
    by Keras Tuner [2].
    
    Parameters
    ----------
    hp: instance of hyperparameters class
        From where hyperparameters will be sampled.
            
    Returns
    -------
    model: Keras sequential model
        
    References
    ----------
    [1] Katzman, Jared L., et al. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." BMC medical research methodology 18.1 (2018): 24.
    [2] https://keras-team.github.io/keras-tuner/
    """
    
    # Construct the (sequential) model.
    model = Sequential()
    
    # model.add(Dense(units=hp['n_features'], activation=hp.Choice('activation_input', ['relu', 'selu']), kernel_initializer='glorot_uniform', input_shape=(hp['n_features'],), name='InputLayer'))
    model.add(Dense(units=7, activation=hp.Choice('activation_input', ['relu', 'selu']), kernel_initializer='glorot_uniform', input_shape=(7,), name='InputLayer'))
    model.add(Dropout(rate=hp.Float('dropout_input', min_value=0.0, max_value=0.75, step=0.05), name='DroputInput'))
    
    
    # for i in range(hp.Int('num_layers', 2, 20)):
    #     model.add(layers.Dense(units=hp.Int('units_' + str(i),
    #                                         min_value=32,
    #                                         max_value=512,
    #                                         step=32),
    #                            activation='relu'))
    # model.add(layers.Dense(10, activation='softmax'))
    # model.compile(
    #     optimizer=keras.optimizers.Adam(
    #         hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy'])
    # return model


    # ###########
    # # Validate inputs.
    # if activation not in ['relu', 'selu']:
    #     raise ValueError(f"{activation} is not a valid activation function.")
        
    # if optimizer not in ['nadam', 'sgd']:
    #     raise ValueError(f"{optimizer} is not a valid optimizer.")
        
        
    # # Construct the (sequential) model.
    # model = Sequential()
    
    # # Input layer.
    # model.add(Dense(units=n_features, activation=activation, kernel_initializer='glorot_uniform', input_shape=(n_features,), name='InputLayer'))
    # model.add(Dropout(dropout, name='DroputInput'))
    
    
    n_layers=2
    n_nodes=25
    activation='relu'
    learning_rate=0.01
    decay=1e-4
    momentum=0.5
    l2_reg=15
    dropout=0.1
    optimizer='nadam'
    
    # Hidden layers are identical between them. 
    # Therefore, we will create them in a loop.
    for n_layer in range(n_layers):
        model.add(Dense(units=n_nodes, activation=activation, kernel_initializer='glorot_uniform', name=f'HiddenLayer{n_layer+1}'))
        model.add(Dropout(dropout, name=f'Dropout{n_layer+1}'))
        
    # Output layer.
    model.add(Dense(units=1, activation='linear', kernel_initializer='glorot_uniform', kernel_regularizer=l2(l2_reg), name='OutputLayer'))
    model.add(ActivityRegularization(l2=l2_reg, name='ActivityRegularization'))
    
    # Define the optimizer
    if optimizer == 'nadam':
        optimizer_ = Nadam(learning_rate=learning_rate, decay=decay)
    elif optimizer == 'sgd':
        optimizer_ = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    
    # Compile the model.
    # Since the loss function is data-dependent, for now we will
    # only use a string as a place holder. Once the model is fit
    # (and the data are available), the proper loss fuction will be defined.
    #model.compile(loss='negative_log_likelihood', optimizer=optimizer_)
    model.compile(loss='mean_squared_error', optimizer=optimizer_)
    
    return model


# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, ActivityRegularization
from tensorflow.keras.optimizers import SGD, Nadam, RMSprop
from tensorflow.keras.regularizers import l2

# %%
import os

tuner = RandomSearch(DeepSurvK_kt, 
                     #objective=kt.Objective("c_index", direction="max"),
                     objective='val_loss',
                     max_trials=5,
                     executions_per_trial=3,
                     directory=os.path.normpath("C:\\"),
                     project_name='05_deepsurvk')

# %%
tuner.search_space_summary()

tuner.search(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test))



# %%

# Retrive best model.
models = tuner.get_best_models(num_models=1)
tuner.results_summary()




# %%
models.fit


# %%
loss = deepsurvk.negative_log_likelihood(E_train)
dsk.compile(loss=loss)

# %%
callbacks = deepsurvk.common_callbacks()

epochs = 500
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


