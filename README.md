# Hands-on data science meetup: Recommender systems in practice

Install using Anaconda or Docker. First clone or download this repository:
```
git clone https://github.com/BigDataRepublic/meetup-ds-recommender-systems.git
cd meetup-ds-recommender-systems
```

## Anaconda
Installation with Anaconda (Python 3.x) environment:
```
conda env create –f environment.yml
jupyter notebook
```

## Docker
Stand-alone installation with Docker.

#### Prerequisits
* Docker Engine 17.09.0+
* Docker Compose 3.4

#### Instructions
Build from source:
```
docker-compose build
docker-compose up
```
Notebook available at `localhost:8889`
Use token in terminal for first-time login.
