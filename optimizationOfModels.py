from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def find_best_params_ElasticNet(X_train, y_train, X_val, y_val, param1_list, param2_list):

    """
    The model selection is based on the mean absolute error
    """

    param1_name = "alpha"
    param2_name = "l1_ratio"

    print("mean_absolute_error [{0},{1}]: mean absolute error".format(param1_name,param2_name))


    scores_mean_absolute_error = np.zeros((len(param1_list), len(param2_list)))
    best_score_mean_absolute_error = float("inf")
    best_parameters = [0, 0]

    for i in range(len(param1_list)):
        for j in range(len(param2_list)):

            param1 = param1_list[i]
            param2 = param2_list[j]

            regressor = ElasticNet(alpha=param1, l1_ratio=param2)
            regressor.fit(X_train, y_train)
            predictions = regressor.predict(X_val)

            mean_absolute_error_ = mean_absolute_error(y_val, predictions)
            print("mean_absolute_error [{0},{1}]: {2}".format(param1, param2, mean_absolute_error_))
            scores_mean_absolute_error[i, j] = mean_absolute_error_

            if mean_absolute_error_ < best_score_mean_absolute_error:
                best_score_mean_absolute_error = mean_absolute_error_
                best_parameters = [param1, param2]

    print("\n-------------------------")
    print("best_mean_absolute_error: {0}".format(best_score_mean_absolute_error))
    print("{0}: {1}".format(param1_name,best_parameters[0]))
    print("{0}: {1}".format(param2_name,best_parameters[1]))
    print("-------------------------\n")

    best_param1 = best_parameters[0]
    best_param2 = best_parameters[1]

    return best_param1, best_param2, scores_mean_absolute_error


def find_best_params_RandomForest(X_train, y_train, X_val, y_val, param1_list, param2_list):

    param1_name = "n_estimators"
    param2_name = "max_depth"

    print("mean_absolute_error [{0},{1}]: mean absolute error".format(param1_name, param2_name))

    scores_mean_absolute_error = np.zeros((len(param1_list), len(param2_list)))
    best_score_mean_absolute_error = float("inf")
    best_parameters = [0, 0]

    for i in range(len(param1_list)):
        for j in range(len(param2_list)):

            param1 = param1_list[i]
            param2 = param2_list[j]

            regressor = RandomForestRegressor(n_estimators=param1,
                                              max_depth=param2,
                                              criterion='mse',
                                              random_state=0,
                                              warm_start = True,
                                              n_jobs=8)

            regressor.fit(X_train, y_train)
            predictions = regressor.predict(X_val)

            mean_absolute_error_ = mean_absolute_error(y_val, predictions)
            print("mean_absolute_error [{0},{1}]: {2}".format(param1, param2, mean_absolute_error_))
            scores_mean_absolute_error[i, j] = mean_absolute_error_

            if mean_absolute_error_ < best_score_mean_absolute_error:
                best_score_mean_absolute_error = mean_absolute_error_
                best_parameters = [param1, param2]

    print("\n-------------------------")
    print("best_mean_absolute_error: {0}".format(best_score_mean_absolute_error))
    print("{0}: {1}".format(param1_name, best_parameters[0]))
    print("{0}: {1}".format(param2_name, best_parameters[1]))
    print("-------------------------\n")

    best_param1 = best_parameters[0]
    best_param2 = best_parameters[1]

    return best_param1, best_param2, scores_mean_absolute_error