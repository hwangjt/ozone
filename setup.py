from numpy.distutils.core import setup, Extension
import os
from subprocess import call


setup(name='ozone',
    version='0.1',
    description='Open-source ODE solver package for gradient-based MDO',
    license='Apache License',
    packages=[
        'ozone',
        'ozone/components',
        'ozone/integrators',
        'ozone/methods',
        'ozone/utils',
    ],
    install_requires=[
    ],
    zip_safe=False,
)
