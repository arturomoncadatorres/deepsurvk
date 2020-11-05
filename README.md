
# DeepSurvK
######  Implementation of DeepSurv using Keras

[![PyPI](https://img.shields.io/pypi/v/deepsurvk.svg)](https://pypi.python.org/pypi/deepsurvk)
[![Build Status](https://img.shields.io/travis/arturomoncadatorres/deepsurvk.svg?branch=master)](https://travis-ci.org/arturomoncadatorres/deepsurvk)
[![Documentation](https://readthedocs.org/projects/deepsurvk/badge/?version=latest)](https://deepsurvk.readthedocs.io/en/latest/?badge=latest)
[![PyUp](https://pyup.io/repos/github/arturomoncadatorres/deepsurvk/shield.svg)](https://pyup.io/repos/github/arturomoncadatorres/deepsurvk/)

DeepSurv is a Cox Proportional Hazards deep neural network used for modeling interactions between a patient's covariates and treatment effectiveness. It was originally proposed by [Katzman et. al (2018)](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1) and [implemented in Theano (using Lasagne)](https://github.com/jaredleekatzman/DeepSurv).

Unfortunately, [Theano is no longer supported](https://groups.google.com/forum/#!msg/theano-users/7Poq8BZutbY/rNCIfvAEAwAJ). There have been some attempts in recreating DeepSurv in other DL platforms, such as [czifan's `DeepSurv.pytorch`](https://github.com/czifan/DeepSurv.pytorch). However, given its popularity and ease of use, I think TensorFlow 2's Keras is a great option for this task.

[mexchy1000 created `DeepSurv_Keras`](https://github.com/mexchy1000/DeepSurv_Keras). However, it is a very raw prototype: it is not properly documented nor validated. Moreover, it is not being actively supported anymore. Therefore, I used it as a rough starting point for the development of DeepSurvK.

This is my first Python package. I am sure there are many places where it could be improved. Feedback is always welcome!

## :bookmark_tabs: Documentation
You can find the complete package's documentation [here](https://deepsurvk.readthedocs.io).

## :tada: Features
* Implemented using Keras (using TensorFlow 2)
* Includes the original datasets together with a proper description of the variables
* Designed with data as pandas DataFrames in mind
* Visualization tools for the most common plots for fast and easy exploration and prototyping
* Treatment recommender
* (Basic) parameter optimization using grid search

## :page_with_curl: License
This package uses the MIT license

## :black_nib: References
If you are using DeepSurvK, please cite the original DeepSurv paper, as well as the current repository as follows:

> * Katzman, Jared L., et al. ["DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network."](https://link.springer.com/article/10.1186/s12874-018-0482-1) BMC medical research methodology 18.1 (2018): 24. [[BibTeX](https://scholar.googleusercontent.com/scholar.bib?q=info:hG13Z0IGDPkJ:scholar.google.com/&output=citation&scisdr=CgXVK4mOEOOa6e7oHyc:AAGBfm0AAAAAXxbtByd6uXB8fbxpWDom9eCJp71TAtUO&scisig=AAGBfm0AAAAAXxbtB35QPVsdnSAHsADGSX408btb6Gvf&scisf=4&ct=citation&cd=-1&hl=en)]
> * Arturo Moncada-Torres. DeepSurvK. Accessed on [MONTH, 20XX].

## :label: Credits
This package was developed in [Spyder](https://www.spyder-ide.org/) (a fantastic open-source Python IDE) using [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [`arturomoncadatorres/cookiecutter-pypackage` project template](https://github.com/arturomoncadatorres/cookiecutter-pypackage).
