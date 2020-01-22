# Interpretability in machine learning for Computational Biology

## Requirements

This course is designed for everyone who would like to learn the basics of interpretability techniques for machine learning. The tutorial will provide a brief introduction to key concepts and recent developments in the field of interpretability.  **Participants should bring a laptop to follow the hands-on exercises.**

### Setup

We provide two main ways to follow the exercises.

#### docker

Assuming you have the image [`drugilsberg/depiction`](https://hub.docker.com/r/drugilsberg/depiction) up-to-date just run:

```console
docker run --mount src=`pwd`/workshops/20200125_AMLD2020/notebooks,target=/workspace/notebooks,type=bind -p 8899:8888 -it drugilsberg/depiction
```

Detailed setup instructions can be found [here](https://github.com/IBM/dl-interpretability-compbio#docker-setup).

#### conda

Instructions can be found [here](https://github.com/IBM/dl-interpretability-compbio#development-setup).

We personally suggest the first setup (docker).

For both the _docker_ and the _conda_ setup, **we strongly suggest to setup your machine prior to the tutorial (ideally the day before)** following the linked instructions. Should you incur into any issue, please do not hesitate to contact [Matteo](mailto:tte@zurich.ibm.com) or [An-phi](mailto:uye@zurich.ibm.com).

## Organisers and tutors

- [Dr. Matteo Manica](https://researcher.watson.ibm.com/researcher/view.php?person=zurich-TTE), IBM Research Zürich
- [An-Phi Nguyen](https://researcher.watson.ibm.com/researcher/view.php?person=zurich-UYE), IBM Research Zürich

## Schedule

| Time        | Title                                | Speaker                  |
|-------------|--------------------------------------|--------------------------|
| 09:00-10:00 | Interpretability in Machine Learning        | An-phi Nguyen  |
| 10:00-10:15 | Introduction to depiction            | Matteo Manica            |
| 10:15-10:30 | (Exercise 1) Hands-on intro to depiction. CellTyper: linear models and interpretability | Matteo Manica           |
| 10:30-11:00 | Coffee break | N/A           |
| 11:00-11:30 | (Exercise 2) Breast cancer image classification  | An-phi Nguyen           |
| 11:30-12:00 | (Exercise 3) Understanding transcription factors binding | An-phi Nguyen          |
| 12:00-12:30 | (Exercise 4) PaccMann: what to do for multimodal data | Matteo Manica           |

## Slides

The slides can be found on Box: https://ibm.box.com/v/amld-2020-depiction.
