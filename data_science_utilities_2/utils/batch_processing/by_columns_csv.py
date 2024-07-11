from os import PathLike
from typing import Union, List
from datetime import datetime
import pandas as pd


class CsvColumnBatchReader:

    def __init__(self, path: PathLike, batch_size: int, header: Union[str, int, List[int]] = 'infer'):
        """
        Creates an object that can be used to read a csv file in column batches instead of traditional row batches.

        This will allow for processing of large csv files that may not fit into memory. It allows for scaling,
        collecting means, and other computationally expensive operations that require the entire column to be in memory.

        :param path: The path to the csv file
        :param batch_size: The number of columns to read at a time
        :param header: Whether the CSV file has a header or not. Default is 'infer'. Refer to pandas.read_csv for more
         information
        """
        self._path = path
        self._batch_size = batch_size
        self._header = header
        self._current_batch_start_index = 0
        self.all_columns = self._get_all_column_names()
        self.current_batch_column_names = []
        self.current_batch_df = pd.DataFrame()
        self.batch_number = 0

    def next(self):
        """
        Reads the next batch of columns from the csv file

        :return: A dataframe of the next batch of columns
        """
        if self.current_batch_df.empty:
            self.current_batch_columns = self.all_columns[: self._batch_size]
        else:
            self.current_batch_columns = self.all_columns[
                self._current_batch_start_index: self._current_batch_start_index + self._batch_size
            ]
            self.batch_number += 1
        print(
            f"Collecting columns index {self._current_batch_start_index} to "
            f"{self._current_batch_start_index + self._batch_size}"
        )
        print(f"Time batch number {self.batch_number} started to be read: {datetime.now().strftime('%H:%M:%S')}")
        self.current_batch_df = pd.read_csv(self._path, usecols=self.current_batch_columns, header=self._header)
        self._current_batch_start_index += self._batch_size
        return self.current_batch_df

    def _get_all_column_names(self):
        """
        Returns the column names from the entire csv file. If no header is present, the column names will be integers
        starting from 0.

        :return: A list of column names
        """
        column_names = pd.read_csv(self._path, nrows=1, header=self._header).columns.tolist()
        print("Collected column names from CSV file")
        return column_names

    def has_next(self):
        """
        Returns whether there are more columns to read from the csv file

        :return: True if there are more columns to read, False otherwise
        """
        return self._current_batch_start_index < len(self.all_columns)
