import random
import pandas as pd


def count_csv_rows_fast(path):
    """
    Counts the number of rows in a CSV file without reading the entire file into memory.

    :param path: The path to the CSV file
    :return: The number of rows in the CSV file
    """
    with open(path, 'r') as file:
        return sum(1 for row in file)


def sample_csv(path, sample_size, verbose=False, **read_csv_kwargs):
    """
    Samples a CSV file without reading the entire file into memory.

    :param path: path to the CSV file
    :param sample_size: the number of rows to sample
    :param verbose: whether to print the progress
    :param read_csv_kwargs: passed to pandas.read_csv
    :return: The sample dataframe
    """
    if sample_size is not None:
        total_rows = count_csv_rows_fast(path)
        skip = sorted(random.sample(range(1, total_rows), total_rows - sample_size - 1))
        sample_df = pd.read_csv(path, skiprows=skip, **read_csv_kwargs)

        if verbose:
            print("Number of rows in base file: ", total_rows)
            print("Number of rows skipped: ", len(skip))
    else:
        sample_df = pd.read_csv(path, **read_csv_kwargs)

    if verbose:
        print("Number of rows in sample: ", sample_df.shape[0])

    return sample_df