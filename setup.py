#!/usr/bin/env python

"""The setup script."""

import setuptools

with open('README.md') as f:
    LONG_DESCRIPTION, LONG_DESC_TYPE = f.read(), 'text/markdown'

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements/base_requirements.txt') as f:
    REQUIREMENTS = f.read().splitlines()

###
# Change values here.
AUTHOR = 'Arturo Moncada-Torres'
AUTHOR_EMAIL = 'arturomoncadatorres@gmail.com'
CLASSIFIERS = ['Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
		'Intended Audience :: Healthcare Industry',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
		'Topic :: Scientific/Engineering :: Artificial Intelligence',
		'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ]
DESCRIPTION = 'Implementation of DeepSurv using Keras'
KEYWORDS = 'deepsurvk',
LICENSE = 'MIT license'
NAME = 'deepsurvk'
PACKAGE_DATA = {'deepsurvk': ['datasets/*']}
PYTHON_REQ = '>=3.7'
SETUP_REQUIREMENTS = ['pytest-runner', 'pandas>=1.0.0']
TEST_REQUIREMENTS = ['pytest>=3', ]
URL = 'https://github.com/arturomoncadatorres/deepsurvk'
###

setuptools.setup(
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    python_requires=PYTHON_REQ,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    entry_points={
        'console_scripts': [
            'deepsurvk=deepsurvk.cli:main',
        ],
    },
    install_requires=REQUIREMENTS,
    license=LICENSE,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    include_package_data=True,
    keywords=KEYWORDS,
    name=NAME,
    packages=setuptools.find_namespace_packages(include=['deepsurvk', 'deepsurvk.*'], exclude=["*.tests", "*.tests.*"]),
    package_data=PACKAGE_DATA,
    setup_requires=SETUP_REQUIREMENTS,
    test_suite='tests',
    tests_require=TEST_REQUIREMENTS,
    url=URL,
    version='0.2.2', # For a complete description of versioning, see https://www.python.org/dev/peps/pep-0440/
    zip_safe=False,
)
