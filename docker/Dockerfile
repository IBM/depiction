FROM continuumio/miniconda3
# labels
LABEL maintainer="Matteo Manica <drugilsberg@gmail.com>, An-phi Nguyen <nguyen.phineas@gmail.com>, Joris Cadow <joriscadow@gmail.com>"
# needed settings
ENV PATH /opt/conda/bin:$PATH
ENV LANG C
# copy the module and setup file
WORKDIR /build/depiction
COPY depiction /build/depiction
WORKDIR /build
COPY setup.py /build/
COPY requirements.txt /build/
# install rdkit
RUN conda config --add channels  https://conda.anaconda.org/rdkit
RUN conda install -y rdkit==2019.03.1
# install pip dependencies
RUN pip install --no-cache-dir -r requirements.txt
# install depiction
RUN pip install --no-cache-dir /build
# install jupyter
RUN pip install --no-cache-dir jupyter==1.0.0
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
ENTRYPOINT ["docker-entrypoint.sh"]