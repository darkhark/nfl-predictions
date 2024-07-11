import numpy as np
import pandas as pd


def get_factor_for_segment_binary_target(segment_mask, df, target_column, pred_col_name):
    """
    Can only be used when the target is binary. The concept of odds as used in this method does not extend naturally to
    multiclass problems. For example, if there are three classes, it's not clear how to calculate the odds of one class
    versus the others.

    Calculates the correction factor for predicted vs actual for the given segment of the population based. This
    requires the target to be binary. The factor is calculated as the ratio of the odds of the target to the odds of the
    prediction. The odds are calculated as the probability of the target divided by 1 minus the probability of the
    target.

    :param segment_mask: Boolean mask for the segment of the population
    :param df: The dataframe containing the probabilities to transform
    :param target_column: The target column name
    :param pred_col_name: The name of the column containing the predictions
    :return: The correction factor
    """
    # If no mask is provided, use the entire dataframe
    if segment_mask is None:
        segment_mask = np.ones(df.shape[0], dtype=bool)
    avg_target_rate = df[segment_mask][[target_column, pred_col_name]].mean()
    target_prob = avg_target_rate[target_column]
    target_pred = avg_target_rate[pred_col_name]

    target_prob_odds = target_prob / (1 - target_prob)
    target_pred_odds = target_pred / (1 - target_pred)
    print("Average target rates (actual and predicted): ", avg_target_rate)
    factor = target_prob_odds / target_pred_odds
    print("\nFactor: ", factor)
    return factor


def adjust_segment_probabilities_binary_target(segment_mask, df, factor, pred_col_name, adj_pred_col_name):
    """
    Can only be used when the target is binary. The concept of odds as used in this method does not extend naturally to
    multiclass problems. For example, if there are three classes, it's not clear how to calculate the odds of one class
    versus the others.

    Adjusts the probabilities for the given segment of the population using the provided factor. Adds two new columns
    to the dataframe: 'target_odds' and the provided 'adj_pred_col_name'.

    :param segment_mask: Boolean mask for the segment of the population
    :param df: The dataframe containing the probabilities to transform
    :param factor: The correction factor for the segment (use get_factor_for_segment_binary_target)
    :param pred_col_name: The name of the column containing the predictions
    :param adj_pred_col_name: The name of the column to store the adjusted probabilities
    :return: None
    """
    print("Factor: ", factor)
    # Calculate the odds of the target for each record
    df['target_odds'] = df[pred_col_name].div(1 - df[pred_col_name])

    # Adjust the probability for the segment using the correction factor.
    df[adj_pred_col_name] = df[pred_col_name].copy()
    # If segment mask is None, adjust all probabilities
    if segment_mask is None:
        segment_mask = np.ones(df.shape[0], dtype=bool)
    # Adjusts the probabilities for the segment to be bound between 0 and 1
    df.loc[segment_mask, adj_pred_col_name] = (df.loc[segment_mask, 'target_odds'] * 1).div(
        (1 + df.loc[segment_mask, 'target_odds'] * factor)
    )


def adjust_grouped_probabilities_binary_target(df, group_col, target_col, pred_col, adj_pred_col, mask=None):
    """
    Can only be used when the target is binary. The concept of odds as used in this method does not extend naturally to
    multiclass problems. For example, if there are three classes, it's not clear how to calculate the odds of one class
    versus the others.

    Adjusts the probabilities for the given segment of the population using the provided factor. Adds two new columns
    to the dataframe: 'target_odds' and the provided 'adj_pred_col_name'.

    :param df: The dataframe containing the probabilities to transform
    :param group_col: The column to group by
    :param target_col: The target column name
    :param pred_col: The name of the column containing the predictions
    :param adj_pred_col: The name of the column to store the adjusted probabilities
    :param mask: The mask to apply to the dataframe before grouping
    :return: The factors for each group
    """
    df['target_odds'] = df[pred_col].div(1 - df[pred_col])

    if mask is None:
        pred_vs_actual = df.groupby(group_col)[[target_col, pred_col]].agg([np.size, np.mean])
    else:
        pred_vs_actual = df[mask].groupby(group_col)[[target_col, pred_col]].agg([np.size, np.mean])

    pred_vs_actual.columns = ['size', target_col, 'size2', 'model_prediction']
    pred_vs_actual = pred_vs_actual.drop(columns='size2')

    actual_odds = pred_vs_actual[target_col].div((1 - pred_vs_actual[target_col]))
    pred_odds = pred_vs_actual['model_prediction'].div((1 - pred_vs_actual['model_prediction']))

    grouped_factors = actual_odds.div(pred_odds)
    df[adj_pred_col] = df[pred_col].copy()

    for unique_val, factor in grouped_factors.reset_index().values:
        val_mask = df[group_col] == unique_val
        df.loc[val_mask, adj_pred_col] = (
                df.loc[val_mask, 'target_odds'] * factor).div((1 + df.loc[val_mask, 'target_odds'] * factor)
        )

    return grouped_factors


def adjust_grouped_probabilities_known_factor_binary_target(df, group_col, pred_col, adj_pred_col, factors: dict):
    """
    Can only be used when the target is binary. The concept of odds as used in this method does not extend naturally to
    multiclass problems. For example, if there are three classes, it's not clear how to calculate the odds of one class
    versus the others.

    Adjusts the probabilities for the given segment of the population using the provided factor. Adds two new columns
    to the dataframe: 'target_odds' and the provided 'adj_pred_col_name'.

    :param df: The dataframe containing the probabilities to transform
    :param group_col: The column to group by
    :param pred_col: The name of the column containing the predictions
    :param factors: The factors for each group
    :return:
    """
    df['target_odds'] = df[pred_col].div(1 - df[pred_col])
    df[adj_pred_col] = df[pred_col].copy()

    for unique_val, factor in factors.items():
        val_mask = df[group_col] == unique_val
        df.loc[val_mask, adj_pred_col] = (
                df.loc[val_mask, 'target_odds'] * factor).div((1 + df.loc[val_mask, 'target_odds'] * factor)
        )

