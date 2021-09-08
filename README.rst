=========================================
TU7 Time in Bed Study: Data Preprocessing
=========================================

This repository contains the documentation on how the TU7 data was preprocessed
to be used in the time in bed analysis and will be most likely the cornerstone
also for further studies working with this data.
The documentation contains code snippets to load the ``.gt3x`` and the data
from the Actiwave ``.edf`` files and store the data for the subjects that worn
both devices in a new HDF5 file ``BEDTIME_TU7.hdf5`` which is compatible with
the pandas API. Later notebooks also show how the manual annotation data can be
added resulting in a new ``ANNOTATED_BEDTIME_TU7.hdf5`` file. All code here was
developed using conda to make it reproducible. To work with this repository
follow the following steps.

Getting Started
===============

Clone this repository
---------------------

.. code-block:: bash

    git clone https://github.com/Trybnetic/sleep-study.git
    cd sleep-study/


Install the dependencies
------------------------

.. code-block:: bash

    conda create -f environment.yml


Activate the environment
------------------------

.. code-block:: bash

    conda activate bedtime


Work on the notebooks
---------------------

.. code-block:: bash

    jupyter notebook


Update the documentation
------------------------

.. code-block:: bash

    make html


Acknowledgements
================

This work was supported by the High North Population Studies at UiT The Arctic
University of Norway.
