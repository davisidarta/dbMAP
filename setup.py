#!/usr/bin/env python

import setuptools

if __name__ == "__main__":
    setuptools.setup()

from dbmap.version import __version__


setuptools.setup(name='dbmap',
      version=__version__,
      packages=setuptools.find_packages(),
      description='dbMAP - fast, accurate and generalized dimensional reduction for explorative data analysis',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      url='https://github.com/davisidarta/dbMAP',
      download_url='https://github.com/davisidarta/dbMAP/archive/1.1.1.tar.gz',
      author='Davi Sidarta-Oliveira',
      author_email='davisidarta@fcm.unicamp.br',
      keywords=['visualization', 'machine-learning', 'dimensionality-reduction', 'topological-data-analysis', 'knowledge-representation','single-cell'],
      license="GNU General Public License v3.0",
      install_requires=[
          'numpy>=1.14.2',
          'pandas>=0.22.0',
          'scipy>=1.0.1',
          'scikit-learn',
          'joblib',
          'tables>=3.4.2',
          'nmslib',
          'annoy',
          'numba'
      ],
      python_requires=">=3",
      )
