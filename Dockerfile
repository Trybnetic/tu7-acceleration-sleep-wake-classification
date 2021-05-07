FROM gpuci/miniconda-cuda

COPY ./environment.yml .
RUN conda create -f environment.yml
RUN conda activate bedtime

CMD jupyter notebook --no-browser --port=8080

EXPOSE 8080
