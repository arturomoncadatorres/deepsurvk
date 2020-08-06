# -*- coding: utf-8 -*-
"""
deepsurvk.py
Defines the actual DeepSurvK network.
"""

import pathlib
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, ActivityRegularization
from tensorflow.keras.optimizers import SGD, Nadam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

from lifelines.utils import concordance_index as c_index

from matplotlib import pyplot as plt

import logzero
from logzero import logger

__all__ = ['DeepSurvK', 'negative_log_likelihood', 'common_callbacks']


#%%
def DeepSurvK(n_features,
              n_layers=2, 
              n_nodes=25, 
              activation='relu', 
              learning_rate=0.01,
              decay=1e-4,
              momentum=0.5,
              l2_reg=15,
              dropout=0.1,
              optimizer='nadam'):
    """
    Create a Keras model using the DeepSurv architecture, as originally
    proposed in [1].
    
    Parameters
    ----------
    n_features: int
        Number of features used by the network.
    n_layers: int
        Number of hidden layers.
        Default is 2.
    n_nodes: int
        Number of nodes of each hidden layer.
        Default is 25.
    activation: string
        Activation function of the input and the hidden layers.
        Possible values are:
            'relu'  Rectified Linear Unit (default)
            'selu'  Scaled exponential linear unit
    learning_rate: float
        Learning rate.
        Default is 0.01.
    decay: float
        Learning rate decay.
        Default is 1e-4
    momentum: float
        Momentum
        Default is 0.5
    l2_reg: float
        L2 regularization
        Default is 15
    droput: float
        Dropout propotion
        Default is 0.1
    optimizer: string
        Model optimizer
        Possible values are:
            'nadam' Nadam (Adam + Nesterov momentum, default) [2]
            'sgd`   Stochastic gradient descent [3]
            
    Returns
    -------
        model: Keras sequential model
        
    References
    ----------
    [1] Katzman, Jared L., et al. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." BMC medical research methodology 18.1 (2018): 24.
    [2] https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Nadam
    [3] https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD
    """
    
    # Validate inputs.
    if activation not in ['relu', 'selu']:
        raise ValueError(f"{activation} is not a valid activation function.")
        
    if optimizer not in ['nadam', 'sgd']:
        raise ValueError(f"{optimizer} is not a valid optimizer.")
        
        
    # Construct the (sequential) model.
    model = Sequential()
    
    # Input layer.
    model.add(Dense(units=n_features, activation=activation, kernel_initializer='glorot_uniform', input_shape=(n_features,), name='InputLayer'))
    model.add(Dropout(dropout, name='DroputInput'))
    
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
    # model.compile(loss=_negative_log_likelihood(E_train), optimizer=optimizer_)
    model.compile(loss='negative_log_likelihood', optimizer=optimizer_)
    
    return model


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


#%%
def common_callbacks():

    callbacks = []
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(f'_temp.h5', monitor='loss', save_best_only=True, mode='min'))

    return callbacks
