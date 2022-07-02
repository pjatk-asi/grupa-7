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

## Conclusions and final notes
Libraries generating prediction models do not handle well string values.
Therefore, we were forced to strip off all string columns in out input data sets.
However, we believe that string colums may affect the accuracy of the model and so those values should be put in dictionaries and then could be referred by index of particular dictionary --- in a way resembling star schema used in data mining.

For evaluation of the generated models we utilized the following metrics:
* Root Main Square Error
* Mean Absolute Error for XGBoost
* R2 Score for Logistic XGBoost

All of above metrics have been applied to both ML algorithms leveraged by us --- i.e. logistic regression and XGBoost.
We used logistic regression only as reference, since we knew from PUM project that XGBoots provides best results for our input data set.