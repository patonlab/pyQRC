from setuptools import setup
import io

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'pyQRC',
  packages = ['pyqrc'],
  version = '1.0.0',
  description = 'A python program for quick (intrinsic) reaction coordinates',
  long_description = long_description,
  long_description_content_type = 'text/markdown',
  author = 'Paton Research Group',
  author_email = 'robert.paton@colostate.edu',
  url = 'https://github.com/bobbypaton/pyQRC',
  download_url = 'https://github.com/bobbypaton/pyQRC/archive/v1.0.0.zip',
  keywords = ['compchem', 'reaction coordinate', 'gaussian', 'IRC'],
  classifiers = [],
  install_requires=[],
  python_requires='>=2.6',
  include_package_data=True,
)
