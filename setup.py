import os
import re
from setuptools import Extension, setup, find_packages
from setuptools.command.install import install
from Cython.Build import cythonize
import numpy

__pkg_name__ = 'bonito'
require_file = 'requirements.txt'
package_name = "ont-%s" % __pkg_name__

verstrline = open(os.path.join(__pkg_name__, '__init__.py'), 'r').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "{}/__init__.py".'.format(__pkg_name__))


with open(require_file) as f:
    requirements = [r.split()[0] for r in f.read().splitlines()]

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()


project_path = os.path.dirname(__file__)
print(os.path.join(project_path,"bonito/hedges_decode/beam_viterbi"))
extensions = [
    Extension("bonito.hedges_decode.context_utils",["bonito/hedges_decode/context_utils/context.pyx"],
              extra_compile_args=["-O3","-std=c++11"],
              include_dirs=[os.path.join(os.getenv("CONDA_PREFIX","/"),"include"),numpy.get_include()],
              library_dirs=[os.path.join(os.getenv("CONDA_PREFIX","/"),"lib")],
              libraries=["hedges_hooks_c"],
              language="c++"
              ),
    Extension("bonito.hedges_decode.beam_viterbi",["bonito/hedges_decode/beam_viterbi/beam_viterbi.pyx"],
              extra_compile_args=["-O3","-std=c++11"],
              include_dirs=[os.path.join(os.getenv("CONDA_PREFIX","/"),"include")],
              runtime_library_dirs=[os.path.join(project_path,"bonito/hedges_decode/beam_viterbi")],
              library_dirs=[os.path.join(os.getenv("CONDA_PREFIX","/"),"lib"),os.path.join(project_path,"bonito/hedges_decode/beam_viterbi")],
              libraries=["hedges_hooks_c","beam_cuda"],
              undef_macros=['NDEBUG'],
              language="c++"
              )
]

setup(
    name=package_name,
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Oxford Nanopore Technologies, Ltd',
    author_email='support@nanoporetech.com',
    url='https://github.com/nanoporetech/bonito',
    entry_points = {
        'console_scripts': [
            '{0} = {0}:main'.format(__pkg_name__)
        ]
    },
    dependency_links=[
        'https://download.pytorch.org/whl/cu113',
    ],
    ext_modules=cythonize(extensions)
)
