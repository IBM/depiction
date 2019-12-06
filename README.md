# depiction

[![Build Status](https://travis-ci.com/IBM/depiction.svg?branch=master)](https://travis-ci.com/IBM/depiction)

A collection of tools and resources to interpret deep learning models in a framework-independent fashion.

The core of the repo is a package, called `depiction`, with wrappers around models and methods for interpretable deep learning.

## Docker setup

### Install docker

Make sure to have a working [docker](https://www.docker.com/) installation.
Installation instructions for different operative systems can be found on the [website](https://docs.docker.com/install/).

### Get `drugilsberg/depiction` image

We built a [docker image](https://cloud.docker.com/repository/docker/drugilsberg/depiction) for `depiction` containing all models, data and dependencies needed to run the notebooks contained in the repo.
Once the docker installation is complete the `depiction` image can be pulled right away:

```sh
docker pull drugilsberg/depiction
```

**NOTE**: the image is quite large (~5.5GB) and this step might require sometime.

### Run `drugilsberg/depiction` image

The image can be run to serve [jupyter](https://jupyter.org/) notebooks by typing:

```sh
docker run -p 8899:8888 -it drugilsberg/depiction
```

At this point just connect to [http://localhost:8899/tree](http://localhost:8899/tree) to run the notebooks and experiment with `depiction`.

#### Daemonization

We recommend to run it as a daemon:

```sh
docker run -d -p 8899:8888 -it drugilsberg/depiction
```

maybe mount your local notebooks directory to keep the changes locally

```
docker run --mount src=`pwd`/notebooks,target=/workspace/notebooks,type=bind -p 8899:8888 -it drugilsberg/depiction
```

and stopped using the container id:

```sh
docker stop <CONTAINER ID>
```

## Development setup

Setup a conda environment

```sh
conda env create -f environment.yml
```

Activate it:

```sh
conda activate depiction-env
```

Install the module:

```sh
pip install .
```
