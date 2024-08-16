============
Introduction
============

Harpia is a framework for parrallel image processing and analysis for 
extremely large images, generated at Sirius synchrotron.  

.. line length------------------------------------------------------------------

Why?
====

Sirius is the Brazilian synchrotron light source. This large equipment 
uses particle accelerators to produce a special type of light called synchrotron 
light, which is used to investigate the composition and structure of matter in 
its most varied forms, with applications in practically all areas of knowledge.

Sirius image beamlines generate data in three dimentions on the order of 
terabytes. In order to analyse the generated data, it is essencial to have high
algorithmic performance for heterogeneous computation systems. Harpia was created
to meet this need, providing high-performance for image segmentation tasks.

Specific Areas We Address
=========================

* Segmentation
* Filters
* Feature extraction
* Images

Where It Works
==============

Harpia is a c++ and cuda framework optimized for nvidia GPU architecture. The 
project also has a python wrapper written in cython, to facilitate 
its use by other projects. Harpia provides the main high-performance features of
Annotat3d project.

Source
======

It's on `GitLab <https://gitlab.cnpem.br/GCD/data-science/segmentation/harpia-model>`_.


Related Publications
--------------------

* `Annotat3D: A Modern Web Application for Interactive Segmentation of Volumetric Images at Sirius/LNLS <https://www.tandfonline.com/doi/full/10.1080/08940886.2022.2112501>`_

