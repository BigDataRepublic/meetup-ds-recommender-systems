FROM continuumio/miniconda3

RUN /opt/conda/bin/conda install jupyter nb_conda -y --quiet

COPY environment.yml /opt

RUN /opt/conda/bin/conda env create -f /opt/environment.yml

RUN mkdir /opt/notebooks
COPY notebooks/bdr-matrix-factorization-recommender.ipynb /opt/notebooks/bdr-matrix-factorization-recommender.ipynb
COPY src /opt/src

ENTRYPOINT ["/bin/sh", "-c"]
