# Opening The Black Box — Interpretability In Deep Learning

## Schedule

| Time | Title | Speaker |
|-------------|--------------------------|---------------|
| 14:00-14:45 | What is interpretability | Matteo Manica |
| 14:45-15:30 | Introducing depiction | Matteo Manica |
| 15:30-16:00 | Coffee break | N/A |
| 16:00-16:45 | Explaining images | Matteo Manica |
| 16:45-17:30 | Explaining tables | Joris Cadow |

## Slides

The slides can be found on box: [https://ibm.box.com/v/odsc-2019-tutorial](https://ibm.box.com/v/odsc-2019-tutorial)

## Running the notebooks

The notebooks can be run either with a conda environment or using docker.  
Either way, follow the general [README.md](../../README.md) and see the respective section below for details concerning the workshop.

### Conda

Depending from where you start the `jupyter notebook` server you might have do minor adjustments to relative paths in notebooks from `workshops/20191120_ODSC2019/notebooks`.

### Docker

With the docker setup we mount a different directory.  
```docker run --mount src=`pwd`/workshops/20191120_ODSC2019/notebooks,target=/workspace/notebooks,type=bind -p 8899:8888 -it drugilsberg/depiction```

and start your browser at [http://localhost:8899/tree/notebooks](http://localhost:8899/tree/notebooks)
