#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as f:
    LONG_DESCRIPTION, LONG_DESC_TYPE = f.read(), "text/markdown"

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Arturo Moncada-Torres",
    author_email='arturomoncadatorres@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Implementation of DeepSurv using Keras",
    entry_points={
        'console_scripts': [
            'deepsurvk=deepsurvk.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    include_package_data=True,
    keywords='deepsurvk',
    name='deepsurvk',
    packages=find_packages(include=['deepsurvk', 'deepsurvk.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/arturomoncadatorres/deepsurvk',
    version='0.1.0',
    zip_safe=False,
)
