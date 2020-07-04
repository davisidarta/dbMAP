import sys
import shutil
from subprocess import call
from setuptools import setup
from warnings import warn

if sys.version_info.major != 3:
    raise RuntimeError('dbMAP requires Python 3')

# get version
with open('src/dbmap/version.py') as f:
    exec(f.read())

# Set README as project description
with open("README.md", "r") as fh:
    long_description = fh.read()


setup(name='dbmap',
      version='1.1.1',
      description='dbMAP - fast, accurate and generalized dimensional reduction for explorative data analysis',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/davisidarta/dbMAP',
      download_url='https://github.com/davisidarta/dbMAP/archive/1.1.1.tar.gz',
      author='Davi Sidarta-Oliveira',
      author_email='davisidarta@gmail.com',
      keywords=['Dimensionality Reduction', 'Big Data', 'Diffusion Maps', 'Nearest-neighbors'],
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
          'scikit-build',
          'nmslib'
      ],
      )
