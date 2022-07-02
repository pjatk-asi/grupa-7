import logging
import os
import pickle
import warnings
from datetime import datetime

import numpy
import pandas
import xgboost
from numpy import sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def save_model(file_name, model):
    file = open(file_name, 'wb')
    pickle.dump(model, file)


def save_evaluation(file_name, datetime, rmse, r2):
    evaluations = pandas.DataFrame(columns=['datetime', 'rmse', 'r2'])
    evaluations = evaluations.append(
        {'datetime': datetime, 'rmse': rmse, 'r2': r2}, ignore_index=True)
    print(evaluations)
    if os.path.isfile(file_name):
        evaluations.to_csv(file_name, index=False, mode='a', header=False)
    else:
        evaluations.to_csv(file_name, index=False)


def hard_drift_check(last_evaluation, other_evaluations):
    return last_evaluation > numpy.mean(other_evaluations)


def parametric_drift_check_rmse(last_evaluation, other_evaluations):
    return last_evaluation > numpy.mean(other_evaluations) + 2 * numpy.std(other_evaluations)


def parametric_drift_check_r2(last_evaluation, other_evaluations):
    return last_evaluation > numpy.mean(other_evaluations) - 2 * numpy.std(other_evaluations)


def check_model(file_name):
    datetime_columns = ['datetime']

    def datetime_parser(input): return datetime.strptime(
        input, '%Y-%m-%d-%H-%M-%S')

    evaluations = pandas.read_csv(
        file_name, parse_dates=datetime_columns, date_parser=datetime_parser)
    last = evaluations['datetime'].max()
    rmse_last = evaluations[evaluations['datetime']
                            == last]['rmse'].values[0]
    rmse_other = evaluations[evaluations['datetime'] != last]['rmse']
    r2_last = evaluations[evaluations['datetime']
                          == last]['r2'].values[0]
    r2_other = evaluations[evaluations['datetime'] != last]['r2']

    rmse_drift = hard_drift_check(rmse_last, rmse_other)
    r2_drift = hard_drift_check(r2_last, r2_other)
    print(f'Root Main Square Error drift: {rmse_drift}, R2 drift: {r2_drift}')

    rmse_parameter_check = parametric_drift_check_rmse(rmse_last, rmse_other)
    r2_parameter_check = parametric_drift_check_r2(r2_last, r2_other)
    print(
        f'Root Main Square Error check: {rmse_parameter_check}, R2 check: {r2_parameter_check}')


def process_csv(csv_uri):
    pandas.set_option('mode.chained_assignment', None)

    # Enable logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)

    # Read data from CSV input file and handle errors if they occur
    csv_uri = 'datasets/training-cleaned-up.csv'

    parse_dates = ['LoanStart', 'LoanEnd']
    dataset = pandas.read_csv(csv_uri, delimiter=';',
                              parse_dates=parse_dates, low_memory=False)

    # Drop redundant columns:
    # (1) LoanNr_ChkDgt --- unique identifier of the loan
    # (2) Drop non-numeric columns --- we could convert the strings into dictionaries
    #     since the values are repeated but we do not know how to do it in python
    # (2) index --- automatically added by pandas library
    dataset.drop('LoanNr_ChkDgt', inplace=True, axis=1)
    dataset.drop('City', inplace=True, axis=1)
    dataset.drop('State', inplace=True, axis=1)
    dataset.drop('Bank', inplace=True, axis=1)
    dataset.drop('BankState', inplace=True, axis=1)
    dataset.drop('Industry', inplace=True, axis=1)
    dataset.drop('LoanStart', inplace=True, axis=1)
    dataset.drop('LoanEnd', inplace=True, axis=1)
    dataset.reset_index(drop=True, inplace=True)

    # Split data set into:
    # (1) 90% --- training dataset
    # (2) 10% --- test dataset
    training, test = train_test_split(dataset, test_size=0.1)

    training = dataset.sample(frac=0.3, random_state=7901)
    test = dataset.drop(training.index)

    # Reshape
    columns = ['Zip', 'NAICS', 'ApprovalFY', 'Term', 'NoEmp', 'Established', 'CreateJob', 'RetainedJob',
               'FranchiseCode', 'UrbanRural',
               'RevLineCr', 'LowDoc', 'BalanceGross', 'LowDocEligible', 'ActiveDuringRecession', 'SBAPortionDecimal',
               'SBA_AppvDecimal', 'GrAppvDecimal',
               'ChgOffPrinGrDecimal', 'DisbursementGrossDecimal']
    target = ['ChargedOff']
    x = training[columns].values.reshape(-1, 20)
    y = training[target].values.reshape(-1, 1)

    # Generate Logistic Regression model
    logistic_regression = LogisticRegression(penalty="l2", tol=0.05, solver="saga")
    logistic_regression.fit(x, y)

    # Generate XGBoost model
    xg_boost = xgboost.XGBClassifier(objective='binary:logistic', colsample_bytree=0.2,
                                     learning_rate=0.02, max_depth=5, alpha=4, n_estimators=1000, min_child_weight=4000)
    xg_boost.fit(x, y)

    # Save generated models
    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logistic_regression_file = 'model/logistic-regression-' + now + '.pkl'
    xgboost_file = 'model/xgboost-' + now + '.pkl'
    save_model(logistic_regression_file, logistic_regression)
    save_model(xgboost_file, xg_boost)

    x_test = test[columns].values.reshape(-1, 20)
    y_test = test[target].values.reshape(-1, 1)

    logistic_regression_predictions = logistic_regression.predict(x_test)
    xg_boost_predictions = xg_boost.predict(x_test)

    # Root Main Square Error for Logistic Regression
    logistic_regression_rmse = sqrt(mean_squared_error(
        y_test, logistic_regression_predictions))
    # R2 Score for Logistic Regression
    logistic_regression_r2 = r2_score(y_test, logistic_regression_predictions)
    print(
        f'Logistic Regression (Root Main Square Error: {logistic_regression_rmse}, R2: {logistic_regression_r2})')

    logistic_regression_evaluation_file_name = 'evaluation/logistic-regression.csv'
    save_evaluation(logistic_regression_evaluation_file_name, now,
                    logistic_regression_rmse, logistic_regression_r2)
    check_model(logistic_regression_evaluation_file_name)

    # Root Main Square Error for Logistic Regression
    xgboost_rmse = sqrt(mean_squared_error(y_test, xg_boost_predictions))
    # R2 Score for Logistic Regression
    xgboost_r2 = r2_score(y_test, xg_boost_predictions)
    print(
        f'XGBoost (Root Main Square Error: {xgboost_rmse}, R2: {xgboost_r2})')

    xgboost_evaluation_file_name = 'evaluation/xgboost.csv'
    save_evaluation(xgboost_evaluation_file_name, now, xgboost_rmse, xgboost_r2)
    check_model(xgboost_evaluation_file_name)
