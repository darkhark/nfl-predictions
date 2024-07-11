import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt


def collect_anomaly_scores_per_column(df, target_column_name):
    """
    Assesses individual data points or observations rather than focusing on a single variable. The scores are calculated
    to identify unusual patterns or outliers in the data, regardless of whether they arise from a single variable or
    multiple variables. By examining the deviation of each observation from the norm, the method can identify anomalies
    across various dimensions.
    :param df: The dataframe to collect the anomaly scores against
    :param target_column_name: The target column name
    :return: A dataframe of the anomaly scores
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target_column_name, axis=1),
        df[target_column_name],
        test_size=0.2,
        random_state=32
    )

    xgb_params = {
        'n_estimators': 50,
        'max_depth': 3,
        'learning_rate': 0.4,
        'random_state': 32
    }

    # Loop through the columns and run the model
    anomaly_scores = []
    for column in X_train.columns:
        X_train_column = X_train.loc[:, column]
        X_test_column = X_test.loc[:, column]

        # Create the model
        model = XGBRegressor(**xgb_params)
        # model.fit(X_train_column.values.reshape(-1, 1), y_train)
        model.fit(X_train_column, y_train)

        # Make predictions
        # preds = model.predict(X_test[column].values.reshape(-1, 1))
        preds = model.predict(X_test_column)

        # Calculate the mean squared error
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Calculate the residuals
        residuals = y_test - preds

        # Calculate the anomaly score
        anomaly_score = np.abs(residuals / np.sqrt(mse))

        # Calculate the mean residual and anomaly score
        mean_residual = residuals.mean()
        mean_anomaly_score = anomaly_score.mean()

        # Append the anomaly score
        record = {
            'column': column,
            'mse': mse,
            'r2': r2,
            'params': xgb_params,
            'mean_residual': mean_residual,
            'mean_anomaly_score': mean_anomaly_score
        }

        anomaly_scores.append(record)

    return pd.DataFrame(anomaly_scores)
