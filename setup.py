import sys
from setuptools import setup, Extension, find_packages
import numpy as np

define_args = '-DBUILD_WITH_PYTHON3' if sys.version.startswith('3') else ''
gsvd_extension = Extension(
                    '_gsvd', 
                    ['src/_gsvd.c'],
                    include_dirs=[
                            np.get_include(), 
                            '/usr/local/include'
                        ],
                    library_dirs=[
                            '/usr/local/lib'
                            ],
                    libraries=['lapacke'],
                    extra_compile_args = [define_args])

setup(
        name='pygsvd',
        version='0.0.2',
        author='Benjamin Naecker',
        author_email='bnaecker@fastmail.com',
        description='Generalized singular value decomposition of NumPy arrays.',
        long_description='''
            The gsvd module is a wrapper for the LAPACK generalized
            singular value routines for double and complex double arrays.
            The GSVD is a joint decomposition, which simultaneously diagonalizes
            a pair of matrices. Intuitively, the decomposition finds a linear
            subspace which most "overlaps" with the row space of the first
            matrix and the null space of the second. The decomposition is useful
            in certain regularization and clustering problems, and can be
            used to derive solutions to the generalized eigenvalue problem.
            ''',
        classifiers = [
                'Development Status :: Alpha',
                'Intended Audience :: Science/Research',
                'Topic :: Scientific/Engineering :: Information Analysis',
                'Licsense :: OSI Approved :: GNU Public License',
                'Operating System :: OS Independent',
                'Programming Language :: Python :: 3'
            ],
        packages=find_packages(),
        ext_modules=[gsvd_extension],
        install_requires=['numpy>=1.11'],
        license='GNU Public License v3',
    )
        
