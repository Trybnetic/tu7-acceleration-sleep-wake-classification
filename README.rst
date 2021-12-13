==========================================================================================================================================
Code and materials from "Discrimination of sleep and wake periods from a hip-worn raw acceleration sensor using recurrent neural networks"
==========================================================================================================================================

This repository contains the documentation, code and materials for our paper 
"Discrimination of sleep and wake periods from a hip-worn raw acceleration 
sensor using recurrent neural networks".
The documentation contains code snippets to load the ``.gt3x`` and the data
from the Actiwave ``.edf`` files and store the data for the subjects that worn
both devices in a new HDF5 file ``BEDTIME_TU7.hdf5`` which is compatible with
the pandas API. Later notebooks also show how the manual annotation data can be
added resulting in a new ``ANNOTATED_BEDTIME_TU7.hdf5`` file. 
Further notebooks can be found in which the performed experiments are documented
and in which the data analysis was conducted.

All code here was developed using conda to make it reproducible. To work with 
this repository follow the following steps.

Getting Started
===============


1. Clone this repository

.. code-block:: bash

    git clone https://github.com/Trybnetic/sleep-study.git
    cd sleep-study/


2. Install the dependencies

.. code-block:: bash

    conda create -f environment.yml


3. Activate the environment

.. code-block:: bash

    conda activate bedtime


4. Work on the notebooks

.. code-block:: bash

    jupyter notebook


Acknowledgements
================

This work was supported by the High North Population Studies at UiT The Arctic
University of Norway.
