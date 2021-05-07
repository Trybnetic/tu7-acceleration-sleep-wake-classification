FROM gpuci/miniconda-cuda:11.1-devel-ubuntu20.04

WORKDIR /home

ADD environment.yml /tmp/environment.yml
RUN conda update -n base -c defaults conda
RUN conda env create -f /tmp/environment.yml

RUN echo "source activate bedtime" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

# CMD jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
ENTRYPOINT ["conda", "run", "-n", "bedtime", "jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
