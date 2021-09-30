FROM gpuci/miniconda-cuda:11.1-devel-ubuntu20.04

WORKDIR /home

ADD environment.yml /tmp/environment.yml
RUN conda update -n base -c defaults conda
RUN conda env create -f /tmp/environment.yml

RUN echo "source activate ml-docs" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

# This is a workaround until paat can be installed from PyPi
ADD pkgs/paat/ pkgs/paat/
WORKDIR pkgs/paat
RUN conda run -n ml-docs pip install -r requirements.txt
RUN conda run -n ml-docs python setup.py develop

WORKDIR /home

# CMD jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
ENTRYPOINT ["conda", "run", "-n", "ml-docs", "jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
