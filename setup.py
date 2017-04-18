#!/usr/bin/env python
# encoding: utf-8

import os
import sys

from setuptools import setup

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/lonpl")
    from lonpl import __version__ as version
except:
    version = "unknown"


if __name__ == '__main__':
    setup(
        name='lonpl',
        author='Daniel Suess',
        author_email='daniel@dsuess.me',
        url='https://github.com/dseuss/lonpl',
        version=version,
        description="Characterising linear optical networks via PhaseLift",
        packages=['lonpl'],
        package_dir={'lonpl': 'lonpl'},
        license="BSD",
        install_requires=['numpy', 'scipy', 'cvxpy', 'autograd', 'six'],
        tests_require=['pytest==3.0.3', 'pytest-warnings'],
        setup_requires=['pytest-runner'],
        test_suite='pytest',
        keywords=[],
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Operating System :: OS Independent",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.3",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
        ],
        platforms=['ALL'],
    )
