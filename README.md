
# DeepSurvK
######  Implementation of DeepSurv using Keras

[![PyPI](https://img.shields.io/pypi/v/deepsurvk.svg)](https://pypi.python.org/pypi/deepsurvk)
[![Build Status](https://img.shields.io/travis/arturomoncadatorres/deepsurvk.svg?branch=master)](https://travis-ci.org/arturomoncadatorres/deepsurvk)
[![Documentation](https://readthedocs.org/projects/deepsurvk/badge/?version=latest)](https://deepsurvk.readthedocs.io/en/latest/?badge=latest)
[![PyUp](https://pyup.io/repos/github/arturomoncadatorres/deepsurvk/shield.svg)](https://pyup.io/repos/github/arturomoncadatorres/deepsurvk/)

DeepSurv is a Cox Proportional Hazards deep neural network used for modeling interactions between a patient's covariates and treatment effectiveness. It was originally proposed by [Katzman et. al (2018)](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1) and [implemented in Theano (using Lasagne)](https://github.com/jaredleekatzman/DeepSurv).

However, [Theano is no longer supported](https://groups.google.com/forum/#!msg/theano-users/7Poq8BZutbY/rNCIfvAEAwAJ). Therefore, I thought it would be a good idea to do my own implementation of it using Keras, which gave birth to DeepSurvK.

There have been a couple of attempts to do so before. For example, [`mexchy1000` has a raw implementation](https://github.com/mexchy1000/DeepSurv_Keras). However, it is not properly documented, it is not validated, and it is not being actively supported anymore. Thus I used that as a rough starting point for the implementation presented here.

This is my first Python package. I am sure there are many places where it could be improved. Feedback is always welcome.

## :bookmark_tabs: Documentation
You can find the complete package's documentation [here](https://deepsurvk.readthedocs.io).

## Features
* TODO

## License
This package uses the MIT license

## :black_nib: References
If you are using DeepSurvK, please cite the original DeepSurv paper, as well as the current repository as follows:

> * Katzman, Jared L., et al. ["DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network."](https://link.springer.com/article/10.1186/s12874-018-0482-1) BMC medical research methodology 18.1 (2018): 24. [[BibTeX](https://scholar.googleusercontent.com/scholar.bib?q=info:hG13Z0IGDPkJ:scholar.google.com/&output=citation&scisdr=CgXVK4mOEOOa6e7oHyc:AAGBfm0AAAAAXxbtByd6uXB8fbxpWDom9eCJp71TAtUO&scisig=AAGBfm0AAAAAXxbtB35QPVsdnSAHsADGSX408btb6Gvf&scisf=4&ct=citation&cd=-1&hl=en)]
> * Arturo Moncada-Torres. DeepSurvK. Accessed on [MONTH, 20XX].

## Credits
This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [`arturomoncadatorres/cookiecutter-pypackage` project template](https://github.com/arturomoncadatorres/cookiecutter-pypackage)
