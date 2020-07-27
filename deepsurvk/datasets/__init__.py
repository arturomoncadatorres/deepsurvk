# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import h5py
from pkg_resources import resource_filename

def _load_dataset(filename, partition='complete', **kwargs):
    """
    Load an example dataset from deepsurvk.datasets.
    All datasets correspond to the ones used originally in [1].
    
    Parameters
    ----------
    filename: string
        Name of the dataset (e.g., 'whas.h5')
    partition: string
        Partition of the data to load. Possible values are:
            'complete' - The whole dataset (default)
            'training' or 'train' - Training partition as used in the original DeepSurv
            'testing' or 'test' - Testing partition as used in the original DeepSurv
    usecols: list
        List of columns in file to use
        
    Returns
    -------
        dataset: pandas DataFrame
        
    References
    ----------
    [1] Katzman, Jared L., et al. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." BMC medical research methodology 18.1 (2018): 24.
    """
    
    filename_ = resource_filename('deepsurvk', 'datasets/data/' + filename)
    
    # Read training data.
    with h5py.File(filename_, 'r') as f:
        X_train = f['train']['x'][()]
        E_train = f['train']['e'][()]
        Y_train = f['train']['t'][()].reshape(-1, 1)
    
    # Read testing data.
    with h5py.File(filename_, 'r') as f:
        X_test = f['test']['x'][()]
        E_test = f['test']['e'][()]
        Y_test = f['test']['t'][()].reshape(-1, 1)
        
    if partition == 'training' or partition == 'train':
        dataset = X_train
    elif partition == 'testing' or partition == 'test':
        dataset = X_test
    elif partition == 'complete':
        dataset = np.concatenate((X_train, X_test), axis=0)
    else:
        raise ValueError('Invalid partition.')

    return dataset


def load_metabric(partition='complete', **kwargs):
    """
    Data from the Molecular Taxonomy of Breast Cancer International
    Consortium (METABRIC), which uses gene and protein expression
    profiles to determine new breast cancer subgroups

    It consists of clinical features of 1980 patients, of which 57.72% 
    have an observed death due to breast cancer with a median survival time 
    of 116 months. However, the file only contains data of 1904 patients.
    
    For more information, see [1] as well as the accompanying README.
    
    References
    ----------
    [1] Curtis, Christina, et al. "The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups." Nature 486.7403 (2012): 346-352.
    """
    return _load_dataset('metabric.h5', partition=partition, **kwargs)


def load_rgbsg(partition='complete', **kwargs):
    """
    The training partition belongs to the Rotterdam tumor bank dataset [1].
    It contains records of 1546 patients with node-positive breast cancer.
    Nearly 90% of the patients have an observed death time. 
    
    The testing partitiong belongs to the German Breast Cancer Study
    Group (GBSG) [2]. It contains records for 686 patients (of which 56 % are
    censored) in a randomized clinical trial that studied the effects of 
    chemotherapy and hormone treatment on survival rate.
    
    For more information, see [1] and [2], as well as the accompanying README.
    
    References
    ----------
    [1] Foekens, John A., et al. "The urokinase system of plasminogen activation and prognosis in 2780 breast cancer patients." Cancer research 60.3 (2000): 636-643.
    [2] Schumacher, M., et al. "Randomized 2 x 2 trial evaluating hormonal treatment and the duration of chemotherapy in node-positive breast cancer patients. German Breast Cancer Study Group." Journal of Clinical Oncology 12.10 (1994): 2086-2093.
    """
    return _load_dataset('rgbsg.h5', partition=partition, **kwargs)


def load_simulated_gaussian(partition='complete', **kwargs):
    """
    Synthetic data with a Gaussian (non-linear) log-risk function.
    
    For more information, see [1] as well as the accompanying README.
    
    References
    ----------
    [1] Katzman, Jared L., et al. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." BMC medical research methodology 18.1 (2018): 24.
    """
    return _load_dataset('simulated_gaussian.h5', partition=partition, **kwargs)


def load_simulated_linear(partition='complete', **kwargs):
    """
    Synthetic data with a linear log-risk function.
    
    For more information, see [1] as well as the accompanying README.
    
    References
    ----------
    [1] Katzman, Jared L., et al. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." BMC medical research methodology 18.1 (2018): 24.
    """
    return _load_dataset('simulated_linear.h5', partition=partition, **kwargs)


def load_simulated_treatment(partition='complete', **kwargs):
    """
    Synthetic data similar to the simulated_gaussian one, with an additional
    column representing treatment.
    
    For more information, see [1] as well as the accompanying README.
    
    References
    ----------
    [1] Katzman, Jared L., et al. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." BMC medical research methodology 18.1 (2018): 24.
    """
    return _load_dataset('simulated_treatment.h5', partition=partition, **kwargs)


def load_support(partition='complete', **kwargs):
    """
    Data from the Study to Understand Prognoses Preferences Outcomes
    and Risks of Treatment (SUPPORT), which studied the survival time of 
    seriously ill hospitalized adults.
    
    Originally, it consists of 14 clinical features of 9105 patients.
    However, patients with missing features were dropped, leaving a total
    of 8873 patients.
    
    For more information, see [1] as well as the accompanying README.
    
    References
    ----------
    [1] Knaus, William A., et al. "The SUPPORT prognostic model: Objective estimates of survival for seriously ill hospitalized adults." Annals of internal medicine 122.3 (1995): 191-203.
    """
    return _load_dataset('support.h5', partition=partition, **kwargs)


def load_whas(partition='complete', **kwargs):
    """
    Data from the Worcester Heart Attack Study (WHAS), which investigates
    the effects of a patient's factors on acute myocardial infraction (MI)
    survival.
    
    It consists of 1638 observations and 5 features: age, sex, 
    body-mass-index (BMI), left heart failure complications (CHF), and order
    of MI (MIORD).
    
    For more information, see [1] as well as the accompanying README.
    
    References
    ----------
    [1] Hosmer Jr, David W., Stanley Lemeshow, and Susanne May. Applied survival analysis: regression modeling of time-to-event data. Vol. 618. John Wiley & Sons, 2011.
    """
    return _load_dataset('whas.h5', partition=partition, **kwargs)