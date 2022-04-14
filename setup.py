# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="IRESNs-tensorflow",
    version="0.2.0",
    description="Library to build ESN, IRESN, IIRESN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SilverLuke/IRESNs",
    author="Luca Argentieri",
    author_email="luca99.argentieri@gmail.com",
    license="GPL3",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    packages=["IRESNs_tensorflow"],
    include_package_data=True,
    install_requires=["numpy", "tensorflow", "keras"]
)
