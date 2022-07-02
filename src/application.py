from process import process_csv


logistic_regression_penalty = 'l2'
logistic_regression_tol = 0.05
logistic_regression_solver = 'saga'
xgboost_colsample_bytree = 0.2
xgboost_learning_rate = 0.02
xgboost_max_depth = 5
xgboost_alpha = 4
xgboost_n_estimators = 1000
xgboost_min_child_width = 4000


process_csv('datasets/training-cleaned-up.csv', logistic_regression_penalty, logistic_regression_tol, logistic_regression_solver,
            xgboost_colsample_bytree, xgboost_learning_rate, xgboost_max_depth, xgboost_alpha, xgboost_n_estimators, xgboost_min_child_width)
process_csv('datasets/test-cleaned-up.csv', logistic_regression_penalty, logistic_regression_tol, logistic_regression_solver,
            xgboost_colsample_bytree, xgboost_learning_rate, xgboost_max_depth, xgboost_alpha, xgboost_n_estimators, xgboost_min_child_width)
process_csv('datasets/production-cleaned-up.csv', logistic_regression_penalty, logistic_regression_tol, logistic_regression_solver,
            xgboost_colsample_bytree, xgboost_learning_rate, xgboost_max_depth, xgboost_alpha, xgboost_n_estimators, xgboost_min_child_width)
process_csv('datasets/validation-cleaned-up.csv', logistic_regression_penalty, logistic_regression_tol, logistic_regression_solver,
            xgboost_colsample_bytree, xgboost_learning_rate, xgboost_max_depth, xgboost_alpha, xgboost_n_estimators, xgboost_min_child_width)
