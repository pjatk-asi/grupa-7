# Prepare environment

## Running application without docker

* Install Anaconda from the website (http://anaconda.com).
* To install create virtual environment run `conda env create -f environment.yaml`
* To run application activate environment `conda activate asi-final-project` and then run `python src/application.py`

## Running application with docker

* Install docker from the website (https://docs.docker.com/get-docker/)
* Pull latest docker container `docker pull lukmal10/asi`
* Run pulled container `docker run lukmal10/asi`

## Dependencies:

* `scikit-learn`
* `pandas`
* `numpy`
* `xgboost`
* `conda`