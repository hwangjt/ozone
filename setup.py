from numpy.distutils.core import setup, Extension
import os
from subprocess import call


setup(name='openode',
    version='0.1',
    description='Open-source ODE solver package for gradient-based MDO',
    license='Apache License',
    packages=[
        'openode',
        'openode/components',
        'openode/integrators',
        'openode/utils',
    ],
    install_requires=[
    ],
    zip_safe=False,
)
