import os
import glob
import numpy
from os.path import join as pjoin
from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

def find_in_path(name, path):
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError(f'The CUDA {k} path could not be located in {v}')
    return cudaconfig

def customize_compiler_for_nvcc(self):
    self.src_extensions.append('.cu')
    default_compiler_so = self.compiler_so
    super = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_so', CUDA['nvcc'])
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']
        super(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so

    self._compile = _compile

class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

CUDA = locate_cuda()

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()
    

filters_cuda_sources = glob.glob('src/filters/**/*.cu', recursive=True)
filters_wrappers = ["harpia/filters/filters.pyx"]

threshold_cuda_sources = glob.glob('src/threshold/**/*.cu', recursive=True)
threshold_wrappers = ["harpia/threshold/threshold.pyx"]

quantification_cuda_sources = glob.glob('src/quantification/**/*.cu', recursive=True)
quantification_wrappers = ["harpia/quantification/quantification.pyx"]

ext_modules = [
    Extension(
        "harpia.filters.filters",
        sources=filters_wrappers + filters_cuda_sources,
        libraries=['cudart'],
        language='c++',
        include_dirs=[numpy_include, CUDA['include'], "include/filters"],
        library_dirs=[CUDA['lib64']],
        runtime_library_dirs=[CUDA['lib64']],
        extra_compile_args={
            'gcc': ['-fPIC'],
            'nvcc': ['--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'", ]
        },
    ),

    Extension(
        "harpia.threshold.threshold",
        sources=threshold_wrappers + threshold_cuda_sources,
        libraries=['cudart'],
        language='c++',
        include_dirs=[numpy_include, CUDA['include'], "include/threshold"],
        library_dirs=[CUDA['lib64']],
        runtime_library_dirs=[CUDA['lib64']],
        extra_compile_args={
            'gcc': ['-fPIC'],
            'nvcc': ['--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'", ]
        },
    ),

    Extension(
        "harpia.quantification.quantification",
        sources=quantification_wrappers + quantification_cuda_sources,
        libraries=['cudart'],
        language='c++',
        include_dirs=[numpy_include, CUDA['include'], "include/quantification"],
        library_dirs=[CUDA['lib64']],
        runtime_library_dirs=[CUDA['lib64']],
        extra_compile_args={
            'gcc': ['-fPIC'],
            'nvcc': ['--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'", ]
        },
    )
]

setup(
    name='cudaext',
    version='0.1',
    description='CUDA extension for Python',
    script_args=["build_ext", "--inplace"],
    ext_modules=cythonize(ext_modules, compiler_directives=dict(language_level="3")),
    cmdclass={'build_ext': custom_build_ext},
    zip_safe=False
)
