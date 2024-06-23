from setuptools import setup
import io

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name='pyqrc',
  packages=['pyqrc'],
  version='1.0.3',
  description='A python program to project computed structures along computed normal modes and perform a Quick Reaction Coordinate calculation',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author='Paton Research Group & pyQRC contributors',
  author_email='patonlab@colostate.edu',
  url='https://github.com/patonlab/pyQRC',
  download_url='https://github.com/patonlab/pyQRC/archive/v1.0.3.zip',
  keywords=['compchem', 'thermochemistry', 'gaussian', 'imaginary frequencies', 'intrinsic reaction coordinate', 'normal modes'],
  classifiers=[],
  install_requires=["cclib"],
  python_requires='>=3.6',
  include_package_data=True,
)
