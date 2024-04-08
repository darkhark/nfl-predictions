import unittest
from unittest.mock import patch, MagicMock
from src.data.next_gen_stats import collect


class TestCollect(unittest.TestCase):

    @patch('nfl_data_py.import_ngs_data')
    def test_get_ngs_data(self, mock_import_ngs_data):
        # Arrange
        mock_import_ngs_data.return_value = MagicMock()
        years = [2018, 2019]
        stat_type = 'passing'

        # Act
        result = collect.get_ngs_data(years, stat_type)

        # Assert
        mock_import_ngs_data.assert_called_once_with(stat_type, years=years)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
