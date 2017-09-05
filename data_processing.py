import pandas as pd
from math import floor

def add_lag(dataframe,
            columns,
            number_of_lag):

    for column in columns:  # for each feature (column)
        for lag in range(number_of_lag):  # for each time step (lag)
            dataframe[column + "_time - " + str(lag)] = dataframe[column].shift(lag)  # copy the previous value

    # remove the first number_of_lag rows
    dataframe.fillna(method='backfill', inplace=True)

    return dataframe


def preprocess_data(data_path,
                    labels_path=None,
                    lag_step_for_SanJuan=10,
                    lag_step_for_Iquitos=10):


    # load data and set index to city, year, weekofyear
    data_frame = pd.read_csv(data_path, index_col=[0, 1, 2])

    features = ['ndvi_ne', 'ndvi_nw',
                'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
                'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
                'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
                'reanalysis_precip_amt_kg_per_m2',
                'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
                'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
                'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
                'station_min_temp_c', 'station_precip_mm']

    data_frame = data_frame[features]

    # fill missing values
    data_frame.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        data_frame = data_frame.join(labels)

    # separate san juan and iquitos
    sanJuan_data = data_frame.loc['sj']
    iquitos_data = data_frame.loc['iq']

    # add lag
    sanJuan_data = add_lag(sanJuan_data, features, lag_step_for_SanJuan)
    iquitos_data = add_lag(iquitos_data, features, lag_step_for_Iquitos)

    # fill navalues
    sanJuan_data.fillna(method='backfill', inplace=True)
    iquitos_data.fillna(method='backfill', inplace=True)

    return sanJuan_data, iquitos_data


def split_time_series(X,
                      y,
                      test_set_ratio):

    number_of_examples = X.shape[0]
    last_index_in_train_set = floor((1 - test_set_ratio) * number_of_examples)

    X_train = X[:last_index_in_train_set]
    X_test = X[last_index_in_train_set:]

    y_train = y[:last_index_in_train_set]
    y_test = y[last_index_in_train_set:]

    return X_train, X_test, y_train, y_test
