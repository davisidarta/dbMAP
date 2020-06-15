import sys
import shutil
from subprocess import call
from setuptools import setup
from warnings import warn

if sys.version_info.major != 3:
    raise RuntimeError('dbMAP requires Python 3')
if sys.version_info.minor < 6:
    warn('Analysis methods were developed using Python 3.6')

# get version
with open('src/dbmap/version.py') as f:
    exec(f.read())

setup(name='dbmap',
      version=__version__,# read in from the exec of version.py; ignore error
      description='dbMAP - a generalized approach for dimensionality reduction aimed at single-cell data.',
      url='https://github.com/davisidarta/dbMAP',
      download_url='https://github.com/davisidarta/dbMAP/archive/1.0.tar.gz',
      author='Davi Sidarta-Oliveira',
      author_email='davisidarta@gmail.com',
      package_dir={'': 'src'},
      packages=['dbmap'],
      install_requires=[
          'numpy>=1.14.2',
          'pandas>=0.22.0',
          'scipy>=1.0.1',
          'networkx>=2.1',
          'scikit-learn',
          'joblib',
          'fcsparser>=0.1.2',
          'tables>=3.4.2',
          'Cython',
          'matplotlib>=2.2.2',
          'seaborn>=0.8.1',
          'tzlocal',
          'scanpy',
          'scikit-build'
      ],
      )
