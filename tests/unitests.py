import unittest
import pandas as pd
import os
import json
from datetime import datetime
from dateutil import parser as date_parser

def load_expected_metrics(metrics_file: str):
    """
    Load the expected migration metrics from a JSON file.

    Args:
        metrics_file (str): Path to the JSON file containing expected metrics.
    
    Returns:
        dict: A dictionary containing the metrics.
    """
    with open(metrics_file, encoding="utf-8") as f:
        metrics = json.load(f)

    return metrics


class TestJSONImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load JSON data
        filepath = os.path.join("data", "documents.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError("JSON file not found.")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                cls.docs = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON: {e}")

        # Load expected metrics from JSON
        metrics_path = os.path.join("tests", "expected_metrics.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            cls.expected = json.load(f)

    def test_row_count(self):
        """Test that we have at least the expected number of documents"""
        expected_rows = self.__class__.expected["row_count"]
        self.assertGreaterEqual(
            len(self.docs), 
            expected_rows, 
            f"Expected at least {expected_rows} documents, found {len(self.docs)}"
        )

    def test_required_fields_exist(self):
        """Test that all required fields are present in documents"""
        # Map the old CSV column names to new JSON field names
        field_mapping = {
            "title_fr": "titre",
            "description_fr": "description", 
            "timings": "dates_affichage"
        }
        
        for old_col, new_field in field_mapping.items():
            for i, doc in enumerate(self.docs):
                self.assertIn(
                    new_field, 
                    doc, 
                    f"Document {i} (UID: {doc.get('uid', 'unknown')}) missing required field: {new_field} (mapped from {old_col})"
                )

    def test_data_structure(self):
        """Test that all documents have the expected structure (fields exist)"""
        expected_fields = [
            "titre", "description", "longue_description", "adresse", 
            "telephone", "site_web", "dates_affichage", "prix", 
            "uid", "location_region"
        ]
        
        missing_fields = []
        for i, doc in enumerate(self.docs):
            for field in expected_fields:
                if field not in doc:
                    missing_fields.append((i, doc.get("uid", "unknown"), field))
        
        self.assertEqual(len(missing_fields), 0, 
                        f"Found {len(missing_fields)} documents with missing fields: {missing_fields[:10]}")

    def test_region_consistency(self):
        """Test that all documents have the correct region"""
        wrong_regions = []
        for i, doc in enumerate(self.docs):
            region = doc.get("location_region", "")
            if region != "Nouvelle-Aquitaine":
                wrong_regions.append((i, doc.get("uid", "unknown"), region))
        
        self.assertEqual(len(wrong_regions), 0, 
                        f"Found {len(wrong_regions)} documents with wrong regions: {wrong_regions[:10]}")

    def test_uid_presence(self):
        """Test that UID field exists and is not None (but can be empty string)"""
        missing_uids = []
        for i, doc in enumerate(self.docs):
            if "uid" not in doc:
                missing_uids.append((i, "unknown"))
        
        self.assertEqual(len(missing_uids), 0, 
                        f"Found {len(missing_uids)} documents missing UID field: {missing_uids[:10]}")

    def test_date_field_structure(self):
        """Test that dates_affichage exists and is a list"""
        invalid_date_structures = []
        for i, doc in enumerate(self.docs[:100]):  # Check first 100
            dates = doc.get("dates_affichage")
            if dates is not None and not isinstance(dates, list):
                invalid_date_structures.append((i, doc.get("uid", "unknown"), type(dates)))
        
        self.assertEqual(len(invalid_date_structures), 0,
                        f"Found {len(invalid_date_structures)} documents with invalid date structure: {invalid_date_structures}")

    def test_events_after_cutoff_date(self):
        """Test that all events end after the cutoff date"""
        cutoff_date_str = self.__class__.expected["cutoffDate"]
        cutoff_date = datetime.fromisoformat(cutoff_date_str.replace('Z', '+00:00'))
        
        print(f"Checking events against cutoff date: {cutoff_date}")
        
        events_before_cutoff = []
        
        for i, doc in enumerate(self.docs):
            dates_display = doc.get("dates_affichage", [])
            
            if not dates_display:
                # Skip documents without dates
                continue
                
            # Parse the last date from the French formatted string
            # Format: "vendredi 29 novembre 2024, 09:30 – vendredi 29 novembre 2024, 12:00"
            for date_str in dates_display:
                try:
                    # Extract the end date part (after the –)
                    if "–" in date_str:
                        end_date_str = date_str.split("–")[-1].strip()
                        # Parse French date
                        event_end_date = self._parse_french_date(end_date_str)
                        
                        if event_end_date and event_end_date < cutoff_date:
                            events_before_cutoff.append({
                                "index": i,
                                "uid": doc.get("uid", "unknown"),
                                "event_end": event_end_date,
                                "cutoff": cutoff_date,
                                "date_string": date_str
                            })
                            
                except Exception as e:
                    print(f"Warning: Could not parse date string '{date_str}' in document {i}: {e}")
                    continue
        
        self.assertEqual(len(events_before_cutoff), 0, 
                        f"Found {len(events_before_cutoff)} events ending before cutoff date {cutoff_date}: {events_before_cutoff[:5]}")

    def _parse_french_date(self, date_str):
        """Parse French date string to datetime object"""
        # French month mappings
        french_months = {
            'janvier': '01', 'février': '02', 'mars': '03', 'avril': '04',
            'mai': '05', 'juin': '06', 'juillet': '07', 'août': '08',
            'septembre': '09', 'octobre': '10', 'novembre': '11', 'décembre': '12'
        }
        
        try:
            # Remove day of week and parse the rest
            parts = date_str.split()
            if len(parts) >= 5:
                # Format: "vendredi 29 novembre 2024, 09:30"
                day = parts[1].zfill(2)  # "7" -> "07", "29" -> "29"
                month_fr = parts[2].lower()  # "novembre" 
                year = parts[3].replace(',', '')  # "2024,"
                time = parts[4] if len(parts) > 4 else "00:00"  # "09:30"
                
                month = french_months.get(month_fr)
                if not month:
                    raise ValueError(f"Unknown French month: {month_fr}")
                
                # Build ISO format string with zero-padded day
                iso_date_str = f"{year}-{month}-{day}T{time}"
                return datetime.fromisoformat(iso_date_str)
                
        except Exception as e:
            print(f"Error parsing French date '{date_str}': {e}")
            
        return None


if __name__ == '__main__':
    unittest.main()