from datetime import datetime
from pyarrow.parquet import ParquetFile
import pandas as pd

class ParquetColumnBatchReader:

    def __init__(self, path, batch_size):
        """
        Creates an object that can be used to read a parquet file in column batches.

        This will allow for processing of large parquet files that may not fit into memory. It allows for scaling,
        collecting means, and other computationally expensive operations that require the entire column to be in memory.

        :param path: The path to the parquet file
        :param batch_size: The number of columns to read at a time
        """
        self._path = path
        self._batch_size = batch_size
        self._current_batch_start_index = 0
        self.all_columns = self._get_all_column_names()
        self.current_batch_column_names = []
        self.current_batch_df = pd.DataFrame()
        self.batch_number = 0

    def next(self):
        """
        Reads the next batch of columns from the parquet file

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
        self.current_batch_df = pd.read_parquet(self._path, columns=self.current_batch_columns)
        self._current_batch_start_index += self._batch_size
        return self.current_batch_df

    def _get_all_column_names(self):
        """
        Returns the column names from the entire parquet file.

        :return: A list of column names
        """
        parquet_file = ParquetFile(self._path)
        column_names = parquet_file.schema.names
        print("Collected column names from parquet file")
        return column_names

    def has_next(self):
        """
        Checks if there are more columns to read

        :return: True if there are more columns to read, False otherwise
        """
        return self._current_batch_start_index < len(self.all_columns)