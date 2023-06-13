import os
import re
from setuptools import Extension, setup, find_packages
from setuptools.command.install import install
from Cython.Build import cythonize

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



extensions = [
    Extension("bonito.hedges_decode.context_utils",["bonito/hedges_decode/context_utils/context.pyx"],
              extra_compile_args=["-O3","-std=c++11"],
              include_dirs=[os.path.join(os.getenv("CONDA_PREFIX","/"),"include")],
              library_dirs=[os.path.join(os.getenv("CONDA_PREFIX","/"),"lib")],
              libraries=["hedges_hooks_c"],
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
