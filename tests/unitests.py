import unittest
import pandas as pd
import os
import json
from datetime import datetime
from utils import load_expected_metrics

class TestCSVImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load CSV
        filepath = os.path.join("output","clean_evenements_publics_openagenda.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError("CSV file not found.")
        try:
            cls.df = pd.read_csv(filepath)
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV: {e}")

        # Load expected metrics from JSON
        metrics_path = os.path.join("tests", "expected_metrics.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            cls.expected = json.load(f)

    def test_row_count(self):
        df = self.__class__.df
        expected_rows = self.__class__.expected["row_count"]
        self.assertEqual(len(df), expected_rows, f"Expected {expected_rows} rows, found {len(df)}")

    def test_required_columns_exist(self):
        df = self.__class__.df
        required_columns = self.__class__.expected["required_columns"]
        for col in required_columns:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

    def test_trade_date_after_threshold(self):
        df = self.__class__.df

        threshold_date = pd.to_datetime(self.__class__.expected["cutoffDate"]).tz_localize("UTC")
    
        column_name = "lastdate_end"
        self.assertIn(column_name, df.columns, f"Missing expected date column: {column_name}")

        try:
            event_dates = pd.to_datetime(df[column_name], errors="raise", utc=True)
        except Exception as e:
            self.fail(f"Error parsing {column_name} column: {e}")

        invalid_dates = event_dates[event_dates <= threshold_date]
        self.assertTrue(invalid_dates.empty, f"Found dates before or on {threshold_date.date()}: {invalid_dates}")


if __name__ == '__main__':
    unittest.main()
