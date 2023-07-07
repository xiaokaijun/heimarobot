#!/usr/bin/env python
import os
import io
import glob
import numpy
import re
import shutil
import sys
from setuptools import setup, find_packages
from distutils.core import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

pypandoc_enabled = True
try:
    import pypandoc
    print('pandoc enabled')
    long_description = pypandoc.convert_file('README.md', 'rst')
except (IOError, ImportError, ModuleNotFoundError):
    print('WARNING: pandoc not enabled')
    long_description = open('README.md',encoding="utf-8").read()
    pypandoc_enabled = False

#import pypandoc
#long_description = pypandoc.convert('README.md', 'rst')
VERSION = find_version('robotpipe', '__init__.py')

requirements = [
    'numpy',
    'onnx',
    'tqdm',
    'requests',
    'matplotlib',
    'Pillow',
    'scipy',
    #'opencv-python',
    'scikit-learn',
    'scikit-image',
    'easydict',
    'cython',
    'albumentations',
    'prettytable',
]

extensions = [
        Extension("robotpipe.insightface.thirdparty.face3d.mesh.cython.mesh_core_cython", 
            ["robotpipe/insightface/thirdparty/face3d/mesh/cython/mesh_core_cython.pyx", "robotpipe/insightface/thirdparty/face3d/mesh/cython/mesh_core.cpp"], language='c++'),
        ]
data_images = list(glob.glob('robotpipe/insightface/data/images/*.jpg'))
data_images += list(glob.glob('robotpipe/insightface/data/images/*.png'))

data_mesh = list(glob.glob('robotpipe/insightface/thirdparty/face3d/mesh/cython/*.h'))
data_mesh += list(glob.glob('robotpipe/insightface/thirdparty/face3d/mesh/cython/*.c'))
data_mesh += list(glob.glob('robotpipe/insightface/thirdparty/face3d/mesh/cython/*.py*'))

data_objects = list(glob.glob('robotpipe/insightface/data/objects/*.pkl'))

data_files = [ ('robotpipe/insightface/data/images', data_images) ]
data_files += [ ('robotpipe/insightface/data/objects', data_objects) ]
data_files += [ ('robotpipe/insightface/thirdparty/face3d/mesh/cython', data_mesh) ]

ext_modules=cythonize(extensions)
setup(
    # Metadata
    name='robotpipe',
    version=VERSION,
    author='robotpipe Contributors',
    author_email='xiaokaijun5027@163.com',
    url='https://github.com/xiaokaijun/robotpipe',
    description='robotpipe Python Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    # Package info
    packages=find_packages(exclude=('docs', 'tests', 'scripts')),
    data_files=data_files,
    zip_safe=True,
    include_package_data=True,
    entry_points={"console_scripts": ["robotpipe-cli=insightface.commands.insightface_cli:main"]},
    install_requires=requirements,
    headers=['robotpipe/insightface/thirdparty/face3d/mesh/cython/mesh_core.h'],
    ext_modules=ext_modules,
    include_dirs=numpy.get_include(),
)

print('pypandoc enabled:', pypandoc_enabled)

