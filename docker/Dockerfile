FROM continuumio/miniconda3
# labels
LABEL maintainer="Matteo Manica <drugilsberg@gmail.com>, An-phi Nguyen <nguyen.phineas@gmail.com>, Joris Cadow <joriscadow@gmail.com>"
# needed settings
ENV PATH /opt/conda/bin:$PATH
ENV LANG C
# install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ python3-dev libxrender-dev\
    && rm -rf /var/lib/apt/lists/*
# install rdkit
RUN conda config --add channels  https://conda.anaconda.org/rdkit
RUN conda install -y rdkit==2019.03.1
# install pip dependencies
WORKDIR /build
COPY requirements.txt /build/
RUN pip install --no-cache-dir -r requirements.txt
# install depiction
WORKDIR /build/depiction
COPY depiction /build/depiction
COPY bin /build/bin
COPY setup.py /build/
RUN pip install --no-cache-dir /build
# install jupyter
RUN pip install --no-cache-dir jupyter==1.0.0
# setup data for tests
WORKDIR /build/data
COPY data /build/data
# setup the workspace
WORKDIR /workspace
COPY data /workspace/data
COPY notebooks /workspace/notebooks
# expose the right port
EXPOSE 8888
# setup the entrypoint
COPY docker/docker-entrypoint.sh /usr/local/bin/
RUN chmod a+x /usr/local/bin/docker-entrypoint.sh
# entrypoint to startup the notebook
CMD ["docker-entrypoint.sh"]
