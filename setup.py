from setuptools import find_packages, setup
from dbmap.version import __version__

setup(name='dbmap',
      version=__version__,
      packages=find_packages(),
      description='dbMAP - fast, accurate and generalized dimensional reduction for explorative data analysis',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      url='https://github.com/davisidarta/dbMAP',
      download_url='https://github.com/davisidarta/dbMAP/archive/1.1.1.tar.gz',
      author='Davi Sidarta-Oliveira',
      author_email='davisidarta@gmail.com',
      keywords=['Dimensionality Reduction', 'Big Data', 'Diffusion Maps', 'Nearest-neighbors'],
      license="GNU General Public License v3.0",
      install_requires=[
          'numpy>=1.14.2',
          'pandas>=0.22.0',
          'scipy>=1.0.1',
          'networkx>=2.1',
          'scikit-learn',
          'joblib',
          'umap-learn',
          'fcsparser>=0.1.2',
          'tables>=3.4.2',
          'nmslib'
      ],
      python_requires=">=3",
      )
