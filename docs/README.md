# Documentation

This repository's documentation is built using Doxygen, Sphinx and Breathe extension.
Both Doxygen and Sphinx are used as documentation generators,
i.e., they read [docstrings](https://en.wikipedia.org/wiki/Docstring) to generate the documentation.

The docstring must follow a pattern for Doxygen and another pattern for Sphinx.
Breathe extension is used as a bridge between Doxygen and Sphinx.

## Installation

First of all, it is necessary to install Doxygen, Sphinx, Read the Docs theme and Breathe. The commands are:

```bash
sudo apt install doxygen
apt-get install python3-sphinx
pip install sphinx-rtd-theme
pip install breathe
```

## Usage

Once the docstrings are written, the documentation is generated using the following commands inside `docs/`:
```bash
sphinx-apidoc -o src/harpia ../
doxygen Doxyfile.in
make
```

For deployment
The folder `build/html/`, containing the documentation, will be created and can be seen on any browser.
```bash
firefox build/html/index.html
```
**Note**: The folder `build/` is untracked by Git because it is too large.

## Configuration

Doxygen is used to autogenerate, in xml format, the documentation for CUDA files in `build/xml/`. Since Doxygen has no support for CUDA, it will be treated as C++.

To do so, the Doxygen configuration file `Doxyfile.in` must have some parameters changed as follows:
```
GENERATE_XML        = YES
XML_OUTPUT          = build/xml     # create xml/ folder inside source/
EXTENSION_MAPPING   = cu=c++
FILE_PATTERNS       = *.cu \
                      *.c \
                      *.cpp \
                      ...
```

The `src/conf.py` is the Sphinx configuration file. It has the autodoc, napoleon, Read the Docs theme and Breathe extensions.
[Autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) is used to write python docstring in the documentation.
[Napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) enables autodoc to read docstrings written in the [Numpy](https://numpydoc.readthedocs.io/en/latest/format.html) or [Google](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) style guide docstrings.
Read the Docs theme extension must be added in order to use Read the Docs theme.
[Breathe](https://breathe.readthedocs.io/en/latest/) reads the files inside `build/xml/` enabling Sphinx to autogenerate documentation from CUDA docstrings, since autodoc can autogenerate documentation only for python.

Then, the `extensions` variable in `conf.py` is set as:
```python
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.mathjax', 'sphinx_rtd_theme', 'breathe']
```

## Tips

We suggest one to use the "Doxygen Documentation Generator" and "Python Docstring Generator" VSCode extensions to type the docstring.

Some useful special commands for Doxygen are `\a` that is used to display the next word in italics. Then it is used to cite parameters. `\f$` has the same meaning as `$` in latex, i.e., it is used to start and finish equations. `\note` is used to write a note.

Also, if one wants to include *LaTeX* equations inside the python docstrings, it is sufficient to add the directive `..math::` before the equation statement.

For more special commands for Doxygen, please check out [this](https://www.doxygen.nl/manual/commands.html). 

- **[Breath documentation - how to write .rst files for c/c++/cuda documentation](https://breathe.readthedocs.io/en/latest/directives.html)**

## Examples of Docstrings

### Doxygen

```
/* header */
/**
 * @file spin_resolution.h
 * @author Matheus L. Bernardi (matheus.bernardi@lnls.br)
 * @brief Resolution assessment routines by Fourier Shell Correlation criteria.
 *
 * @details \f$ F_{SC}(r)=\frac{\sum_{r_i\in r} F_1(r_i)\cdot F_2(r_i)^*}{\sqrt{\sum_{r_i\in r}|F_1(r_i)|^2\cdot\sum_{r_i\in r}|F_2(r_i)|^2}}\f$
 *
 * @date 2022-01-07
 * @see https://www.sciencedirect.com/science/article/abs/pii/S1047847705001292
 */
 
/**
 * @brief Computes Fourier Ring Correlation (FRC) curves
 *
 * @param image1
 * @param image2
 * @param out_numreal Real numerator of FRC curve
 * @param out_numimag Imaginary numerator of FRC curve
 * @param out_den1 Squared module of Fourier transform of image1 \f$ |\mathcal{F}(I_1)|^2\f$
 * @param out_den2 Squared module of Fourier transform of image2 \f$ |\mathcal{F}(I_2)|^2\f$
 * @param out_pixelcounter Array containing the counting of pixels per ring
 * @param xsize
 * @param ysize
 * @param zsize
 */
void spin_fourier_ring_correlation(cufftComplex *image1, cufftComplex *image2, float *out_numreal, float *out_numimag,
                                   float *out_den1, float *out_den2, int *out_pixelcounter, int xsize, int ysize,
                                   int zsize);
```

### Sphinx

Basic example with *LaTeX*:

```python
def spin_get_fourier_correlation_sigma_threshold(pixelcounter):
    r""" Get 3-sigma threshold curve

    Calculates the 3-sigma threshold criterion for ``FSC`` resolution assessment.

    The curve is defined as:

    .. math::
        \sigma(r)=\frac{\sigma_{\text{factor}}}{\sqrt{n(r)/2}}\cdot\sqrt{n_{\text{asym}}}

    Args:
        pixelcounter (array): array with number of pixels between each ring/shell

    Returns:
        array: 3-sigma curve array
    """
    
    # MAGIC HAPPENS!

    return
```

Example using Notes & References to cite stuff:

```python
def spin_gpu_fourier_edt(binary, lbda, method, delta=0.01):
    """ Computes Euclidean Distance Transform w/ FFT

    A first approach to implementing Distance Transforms using FFT [1]_ on CUDA

    Args:
        binary (array): binary img
        lbda (double): lambda parameter
        method (enum): choice between log-convolution and diff-convolution theorems
        delta (double): step for finite difference

    Returns:
        (array): EDT transform of binary.

    Notes:
        For detailed information on the methods, please see https://ieeexplore.ieee.org/document/8686167

    References:
        .. [1] C. Karam, K. Sugimoto and K. Hirakawa, "Fast Convolutional Distance Transform," in IEEE Signal Processing Letters, vol. 26, no. 6, pp. 853-857, June 2019, doi: 10.1109/LSP.2019.2910466.
    """

    # POOF!

    return
```

Example with multiple returns:

```python
def spin_interpolated_intercept(x, y1, y2):
    """Find the intercept of two curves, given by the same x data

    Args:
        x (array): x coordinate array
        y1 (array): curve1 array
        y2 (array): curve2 array

    Returns:
        (array, array): array with x interception values, array with y interception values
    """

    # WOOOOOOOOOOOOOOW

    return
```
