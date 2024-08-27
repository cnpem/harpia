import glob
import os
from os.path import join as pjoin

import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import setup
from setuptools.extension import Extension

import versioneer

# Read the version from __version__.py
# Cannot use 'from harpia import __version__' because other references are not found
# before Cython compilation (e.g., cannot find 'harpia.filters.filters' and others).
version = {}
with open(os.path.join("harpia", "__version__.py")) as fp:
    exec(fp.read(), version)


def find_in_path(name, path):
    """Find a file in a search path.

    Parameters:
        name (str): The name of the file to find.
        path (str): The search path (e.g., PATH environment variable).

    Returns:
        str: The absolute path to the file if found, otherwise None.
    """

    # Adapted from http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system.

    Returns:
        dict: A dictionary with keys 'home', 'nvcc', 'include', and 'lib64',
              containing the absolute paths to each directory.

    Raises:
        EnvironmentError: If the CUDA environment cannot be found or paths are missing.
    """

    # First check if the CUDAHOME environment variable is in use
    if "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
        nvcc = pjoin(home, "bin", "nvcc")
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            raise EnvironmentError(
                "The nvcc binary could not be located in your $PATH. " "Either add it to your path, or set $CUDAHOME."
            )
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": pjoin(home, "include"),
        "lib64": pjoin(home, "lib64"),
    }
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError(f"The CUDA {k} path could not be located in {v}")

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """Customize the compiler for handling CUDA source files.

    This function injects deep into distutils to customize how the dispatch
    to gcc/nvcc works. It allows the compiler to process .cu files and
    redefines the _compile method to use nvcc for .cu files.

    Notes:
        - The method modifies the compiler behavior based on the file extension.
        - It restores the default compiler after compiling CUDA files.
    """

    # Tell the compiler it can process .cu files
    self.src_extensions.append(".cu")

    # Save references to the default compiler_so and _compile methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        print("extra_postargs:", extra_postargs)
        if os.path.splitext(src)[1] == ".cu":
            # Use the CUDA compiler for .cu files
            self.set_executable("compiler_so", CUDA["nvcc"])
            # Use only a subset of the extra_postargs for nvcc
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so after CUDA compilation
        self.compiler_so = default_compiler_so

    # Inject the redefined _compile method into the class
    self._compile = _compile


# Run the customize_compiler function
class custom_build_ext(build_ext):
    def build_extensions(self):
        """Customize and build the extensions using the nvcc compiler for CUDA files."""
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


# Global variables to be used by get_extension_modules()
CUDA = locate_cuda()

cuda_sources = []
# Collect CUDA source files from the 'src' directory
for root, _, files in os.walk("src"):
    for file in files:
        if file.endswith(".cu"):
            cuda_sources.append(os.path.join(root, file))

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


def get_extension_modules(basedir):
    """Generate a list of extension modules for Cython compilation, including CUDA sources.

    Parameters:
        basedir (str): The base directory to search for .py and .pyx files.

    Returns:
        list: A list of setuptools.Extension objects.

    The function performs the following steps:
    1. Collects all .py and .pyx files in the specified base directory.
    2. Defines a helper function `_ext_name` to construct the extension name based on the file path.
    3. Creates a list of Extension objects for each file, configuring them with
       necessary CUDA settings and other compilation parameters.

    Notes:
        - The `cuda_sources` variable should be defined elsewhere in your script.
        - The `CUDA` dictionary should contain paths for 'include' and 'lib64' directories.
        - The `numpy_include` variable should be defined elsewhere in your script.
    """

    files = [
        *glob.glob(os.path.join(basedir.replace(".", os.path.sep), "*.py")),
        *glob.glob(os.path.join(basedir.replace(".", os.path.sep), "*.pyx")),
    ]

    def _ext_name(basedir, file):
        """Construct the extension name based on the file path."""
        name = basedir + "." + os.path.splitext(os.path.basename(file))[0]
        return name

    ext = [
        Extension(
            _ext_name(basedir, file),
            sources=cuda_sources + [file],
            libraries=["cudart"],
            language="c++",
            include_dirs=[numpy_include, CUDA["include"], "src"],
            library_dirs=[CUDA["lib64"]],
            runtime_library_dirs=[CUDA["lib64"]],
            extra_compile_args={
                "gcc": [],
                "nvcc": ["--ptxas-options=-v", "-c", "--compiler-options", "'-fPIC'"],
            },
        )
        for file in files
    ]
    print(ext)
    return ext


# Create Extension objects for various submodules
ext_modules = [
    *get_extension_modules("harpia"),
    *get_extension_modules("harpia.morphology"),
    *get_extension_modules("harpia.filters"),
    *get_extension_modules("harpia.quantification"),
    *get_extension_modules("harpia.threshold"),
]

print(cuda_sources)
print(files)
print(ext_modules)
setup(
    name="harpia",
    version=versioneer.get_version(),
    #   cmdclass=versioneer.get_cmdclass(),
    #   version=version['__version__'],
    description="CUDA extension for Python",
    script_args=["build_ext", "--inplace", "bdist_wheel"],
    ext_modules=cythonize(
        ext_modules,
        compiler_directives=dict(
            language_level="3",
        ),
    ),
    cmdclass={"build_ext": custom_build_ext},
    zip_safe=False,
)
