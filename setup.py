import os
import glob
import numpy
from os.path import join as pjoin
from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


# Read the version from __version__.py
# I cant use from harpia import __version__ because other references are not found before cython compilation
# i.e cannot find harpia.filters.filters and so on
version = {}
with open(os.path.join("harpia", "__version__.py")) as fp:
    exec(fp.read(), version)

def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, '
                'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        print('extra_postargs:',extra_postargs)
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile


# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

# Global variables to be used by get_extension_modules()
CUDA = locate_cuda()

cuda_sources = []
# TODO: correct some compilation bugs for filters and use 'src' folder
#for root, _, files in os.walk('src'): 
for root, _, files in os.walk('src'):
     for file in files:
        if file.endswith('.cu'):
            cuda_sources.append(os.path.join(root, file))

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()
    

def get_extension_modules(basedir):
    """
    Generates a list of extension modules for Cython compilation, including CUDA sources.

    Args:
        basedir (str): The base directory to search for .py and .pyx files.

    Returns:
        list: A list of setuptools.Extension objects.

    The function performs the following steps:
    1. Collects all .py and .pyx files in the specified base directory.
    2. Defines a helper function `_ext_name` to construct the extension name based on the file path.
    3. Creates a list of `setuptools.Extension` objects for each file, configuring them with
       necessary CUDA settings and other compilation parameters.

    Notes:
        - 
        - The `cuda_sources` variable should be defined elsewhere in your script.
        - The `CUDA` dictionary should contain paths for 'include' and 'lib64' directories.
        - The `numpy_include` variable should be defined elsewhere in your script.
    """

    files = [
        *glob.glob(os.path.join(basedir.replace('.', os.path.sep), '*.py')),
        *glob.glob(os.path.join(basedir.replace('.', os.path.sep), '*.pyx'))
    ]

    def _ext_name(basedir, file):
        name = basedir + '.' + os.path.splitext(os.path.basename(file))[0]
        return name


    ext = [
        Extension(
        _ext_name(basedir, file),
        sources=cuda_sources + [file],
        libraries=['cudart'],
        language='c++',
        include_dirs=[numpy_include, CUDA['include'], "src"],
        library_dirs=[CUDA['lib64']],
        runtime_library_dirs=[CUDA['lib64']],
        extra_compile_args={
            'gcc': [],
            'nvcc': ['--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"]
        }) for file in files
    ]
    print(ext)
    return ext

# Create Extension objects
ext_modules = [
    *get_extension_modules('harpia'), 
    *get_extension_modules('harpia.morphology'),
    *get_extension_modules('harpia.filters'),
    *get_extension_modules('harpia.quantification'),
    *get_extension_modules('harpia.threshold')
]

print(cuda_sources)
print(files)
print(ext_modules)
setup(
    name='harpia',
    version=version['__version__'],
    description='CUDA extension for Python',
    script_args = ["build_ext", "--inplace", "bdist_wheel"],
    ext_modules=cythonize(
        ext_modules,
        compiler_directives=dict(
            language_level="3",
        )
    ),
    cmdclass={'build_ext': custom_build_ext},
    zip_safe=False
)
