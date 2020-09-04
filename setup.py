#!/usr/bin/env python

"""The setup script."""

import setuptools

# For a complete description of versioning, see https://www.python.org/dev/peps/pep-0440/
# exec(compile(open("deepsurvk/version.py").read(), "deepsurvk/version.py", "exec"))
#__version__ = 'version='0.1.1''

with open("README.md") as f:
    LONG_DESCRIPTION, LONG_DESC_TYPE = f.read(), "text/markdown"

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open("requirements/base_requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

setup_requirements = ['pytest-runner', 'pandas>=1.0.0']

test_requirements = ['pytest>=3', ]

setuptools.setup(
    author="Arturo Moncada-Torres",
    author_email='arturomoncadatorres@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
		'Intended Audience :: Healthcare Industry',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
		'Topic :: Scientific/Engineering :: Artificial Intelligence',
		'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
    description="Implementation of DeepSurv using Keras",
    entry_points={
        'console_scripts': [
            'deepsurvk=deepsurvk.cli:main',
        ],
    },
    install_requires=REQUIREMENTS,
    license="MIT license",
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    include_package_data=True,
    keywords='deepsurvk',
    name='deepsurvk',
    packages=setuptools.find_packages(include=['deepsurvk', 'deepsurvk.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/arturomoncadatorres/deepsurvk',
    version='0.1.0',
    zip_safe=False,
)
