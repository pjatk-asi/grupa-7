from sys import argv
from process import process_csv
import mlflow
import mlflow.sklearn


def train_and_log(file_path, logistic_regression_penalty, logistic_regression_tol, logistic_regression_solver, xgboost_colsample_bytree, xgboost_learning_rate, xgboost_max_depth, xgboost_alpha, xgboost_n_estimators, xgboost_min_child_width):
    with mlflow.start_run():
        (logistic_regression, logistic_regression_rmse, logistic_regression_mae, logistic_regression_r2, xgboost, xgboost_rmse, xgboost_mae, xgboost_r2) = process_csv(file_path, logistic_regression_penalty, logistic_regression_tol, logistic_regression_solver,
                                                                                                                                                                       xgboost_colsample_bytree, xgboost_learning_rate, xgboost_max_depth, xgboost_alpha, xgboost_n_estimators, xgboost_min_child_width)
        mlflow.log_param('logistic_regression_penalty',
                         logistic_regression_penalty)
        mlflow.log_param('logistic_regression_tol', logistic_regression_tol)
        mlflow.log_param('logistic_regression_solver',
                         logistic_regression_solver)
        mlflow.log_param('xgboost_colsample_bytree', xgboost_colsample_bytree)
        mlflow.log_param('xgboost_learning_rate', xgboost_learning_rate)
        mlflow.log_param('xgboost_max_depth', xgboost_max_depth)
        mlflow.log_param('xgboost_alpha', xgboost_alpha)
        mlflow.log_param('xgboost_n_estimators', xgboost_n_estimators)
        mlflow.log_param('xgboost_min_child_width', xgboost_min_child_width)
        mlflow.log_metric('logistic_regression_rmse', logistic_regression_rmse)
        mlflow.log_metric('logistic_regression_mae', logistic_regression_mae)
        mlflow.log_metric('logistic_regression_r2', logistic_regression_r2)
        mlflow.log_metric('xgboost_rmse', xgboost_rmse)
        mlflow.log_metric('xgboost_mae', xgboost_mae)
        mlflow.log_metric('xgboost_r2', xgboost_r2)
        mlflow.sklearn.log_model(logistic_regression, 'logistic_regression')
        mlflow.sklearn.log_model(xgboost, 'xgboost')


logistic_regression_penalty = argv[1] if len(argv) > 1 else 'l2'
logistic_regression_tol = float(argv[2]) if len(argv) > 2 else 0.05
logistic_regression_solver = argv[3] if len(argv) > 3 else 'saga'
xgboost_colsample_bytree = float(argv[4]) if len(argv) > 4 else 0.2
xgboost_learning_rate = float(argv[5]) if len(argv) > 5 else 0.02
xgboost_max_depth = int(argv[6]) if len(argv) > 6 else 5
xgboost_alpha = int(argv(7)) if len(argv) > 7 else 4
xgboost_n_estimators = int(argv[8]) if len(argv) > 8 else 1000
xgboost_min_child_width = int(argv[9]) if len(argv) > 9 else 4000


train_and_log('datasets/training-cleaned-up.csv', logistic_regression_penalty, logistic_regression_tol, logistic_regression_solver,
              xgboost_colsample_bytree, xgboost_learning_rate, xgboost_max_depth, xgboost_alpha, xgboost_n_estimators, xgboost_min_child_width)
train_and_log('datasets/test-cleaned-up.csv', logistic_regression_penalty, logistic_regression_tol, logistic_regression_solver,
              xgboost_colsample_bytree, xgboost_learning_rate, xgboost_max_depth, xgboost_alpha, xgboost_n_estimators, xgboost_min_child_width)
train_and_log('datasets/production-cleaned-up.csv', logistic_regression_penalty, logistic_regression_tol, logistic_regression_solver,
              xgboost_colsample_bytree, xgboost_learning_rate, xgboost_max_depth, xgboost_alpha, xgboost_n_estimators, xgboost_min_child_width)
train_and_log('datasets/validation-cleaned-up.csv', logistic_regression_penalty, logistic_regression_tol, logistic_regression_solver,
              xgboost_colsample_bytree, xgboost_learning_rate, xgboost_max_depth, xgboost_alpha, xgboost_n_estimators, xgboost_min_child_width)
