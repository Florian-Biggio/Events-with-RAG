import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from babel.dates import format_datetime
from bs4 import BeautifulSoup
import warnings
from bs4 import MarkupResemblesLocatorWarning
import time
from tqdm import tqdm
import argparse

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ===== CONFIGURATION =====
CUTOFF_DATE = datetime.now().strftime("%Y-%m-%dT00:00:00")
CUTOFF_DATE = CUTOFF_DATE[0:3] + str(int(CUTOFF_DATE[3])-1) + CUTOFF_DATE[4:] # today but one year ago

REGION = "Nouvelle-Aquitaine"
DEFAULT_MAX_DOCUMENTS = 500
DEFAULT_DATA_PATH = "imported_events_data"
# =========================

def fetch_events_data(cutoff_date, region, max_documents=20000):
    """Fetch events using comprehensive 3-month ranges from 1 year ago to 3+ years future"""
    print("Fetching events data...")
    
    base_url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
    all_records = []
    
    # Parse dates
    cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%dT%H:%M:%S")
    today = datetime.now()
    
    # Calculate start date (1 year ago from today or cutoff, whichever is earlier)
    one_year_ago = today - timedelta(days=365)
    start_date = min(cutoff_dt, one_year_ago)
    
    date_ranges = []
    
    # 1 year in past + 3 years in future = 4 years = 16 quarters
    for i in range(16):
        range_start = start_date + timedelta(days=90 * i)
        
        # For the last range, no end date to get all remaining future events
        if i == 15:
            date_ranges.append((
                range_start.strftime("%Y-%m-%d"),
                None  # All future events beyond our 4-year coverage
            ))
        else:
            range_end = range_start + timedelta(days=90) - timedelta(days=1)  # End day before next range starts
            date_ranges.append((
                range_start.strftime("%Y-%m-%d"),
                range_end.strftime("%Y-%m-%d")
            ))
    
    print(f"Using {len(date_ranges)} comprehensive 3-month date ranges")
    print(f"Covering from {date_ranges[0][0]} to all future events")
    print(f"Total time span: 1 year past + 3 years future = 4 years")
    
    for i, (start_date, end_date) in enumerate(date_ranges):
        if end_date:
            print(f"[{i+1}/{len(date_ranges)}] Fetching events from {start_date} to {end_date}...")
            date_filter = f"firstdate_begin >= '{start_date}' AND firstdate_begin <= '{end_date}'"
        else:
            print(f"[{i+1}/{len(date_ranges)}] Fetching events from {start_date} onwards...")
            date_filter = f"firstdate_begin >= '{start_date}'"
        
        offset = 0
        range_records = []
        
        while offset < 10000:  # Stay under API offset limit
            params = {
                "limit": 100,
                "where": f"location_region = '{region}' AND lastdate_end >= '{cutoff_date}' AND {date_filter}",
                "offset": offset,
            }
            
            try:
                response = requests.get(base_url, params=params)
                
                if response.status_code == 400:
                    print(f"  API limit reached at offset {offset}")
                    break
                    
                response.raise_for_status()
                data = response.json()
                records = data.get("results", [])
                
                if not records:
                    break
                    
                range_records.extend(records)
                offset += len(records)
                
                print(f"  Fetched {len(records)} records (total in range: {len(range_records)})")
                    
            except requests.exceptions.HTTPError as e:
                print(f"  Error: {e}")
                break
            
            time.sleep(0.02)
        
        all_records.extend(range_records)
        date_range_desc = f"{start_date} to {end_date}" if end_date else f"{start_date} onwards"
        print(f"Completed: {len(range_records)} records from {date_range_desc}")
        
        # Early exit if we've reached our target
        if len(all_records) >= max_documents:
            all_records = all_records[:max_documents]
            print(f"Reached maximum document limit of {max_documents}")
            break
    
    print(f"Final total records fetched: {len(all_records)}")
    
    # Optional: Show date distribution of fetched records
    if all_records:
        df_temp = pd.DataFrame.from_records(all_records)
        if 'firstdate_begin' in df_temp.columns:
            df_temp['firstdate_begin'] = pd.to_datetime(df_temp['firstdate_begin'])
            print(f"Date range of fetched events: {df_temp['firstdate_begin'].min().date()} to {df_temp['firstdate_begin'].max().date()}")
    
    return pd.DataFrame.from_records(all_records)

def clean_html(text):
    """Clean HTML from text fields"""
    if isinstance(text, str):
        return BeautifulSoup(text, "html.parser").get_text(separator="")
    return ""

def parse_dates(dates_str):
    """Parse date strings into structured format"""
    try:
        parsed = json.loads(dates_str)
        return [
            {
                "begin": date_parser.parse(item["begin"]),  
                "end": date_parser.parse(item["end"])       
            }
            for item in parsed
        ]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []

def format_dates_fr(dates):
    """Format dates in French locale"""
    return [
        f"{format_datetime(d['begin'], 'EEEE d MMMM yyyy, HH:mm', locale='fr_FR')} – {format_datetime(d['end'], 'EEEE d MMMM yyyy, HH:mm', locale='fr_FR')}"
        for d in dates
    ]

def build_documents_from_df(df):
    """Convert DataFrame to document format"""
    print("Building documents...")
    
    documents = []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Building documents")):
        raw_dates = row.get("timings", "")
        parsed_dates = parse_dates(raw_dates)
        display_dates = format_dates_fr(parsed_dates)
        
        doc = {
            "titre": row.get("title_fr", "") or "",
            "description": row.get("description_fr", "") or "",
            "longue_description": clean_html(row.get("longdescription_fr", "") or ""),
            "adresse": ", ".join(filter(None, [
                row.get("location_address", "")
            ])),
            "telephone": row.get("location_phone", "") or "",
            "site_web": row.get("location_website", "") or "",
            # Only include the display dates (strings), not the datetime objects
            "dates_affichage": display_dates,
            "prix": row.get("conditions_fr", "") or "",
            "uid": row.get("uid", ""),
            "location_region": row.get("location_region", ""),
        }
        documents.append(doc)
    
    print(f"Built {len(documents)} documents")
    return documents

def save_imported_data(docs, data_path):
    """Save imported documents to JSON file"""
    os.makedirs(data_path, exist_ok=True)
    
    # Create a serializable version without datetime objects
    serializable_docs = []
    for doc in docs:
        serializable_doc = doc.copy()
        # Remove the dates field that contains datetime objects
        # Keep only the display dates which are already strings
        if "dates" in serializable_doc:
            del serializable_doc["dates"]
        serializable_docs.append(serializable_doc)
    
    # Save documents
    with open(os.path.join(data_path, "documents.json"), "w", encoding="utf-8") as f:
        json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
    
    # Save metadata
    metadata = {
        "cutoff_date": CUTOFF_DATE,
        "region": REGION,
        "document_count": len(docs),
        "import_date": datetime.now().isoformat(),
    }
    
    with open(os.path.join(data_path, "import_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Data saved to: {data_path}")
    print(f"Documents: {len(docs)}")
    return data_path

def main(data_path=DEFAULT_DATA_PATH, max_documents=DEFAULT_MAX_DOCUMENTS):
    print("Starting data import...")
    print(f"Configuration: {max_documents} documents, region: {REGION}")
    print(f"Data path: {data_path}")
    
    try:
        # Step 1: Fetch data
        df = fetch_events_data(CUTOFF_DATE, REGION, max_documents) 
        
        # Step 2: Convert to documents
        docs = build_documents_from_df(df)
        
        # Step 3: Save imported data
        save_path = save_imported_data(docs, data_path)
        
        print("Data import completed successfully!")
        return docs
        
    except Exception as e:
        print(f"Error in data import: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Import events data from API')
    parser.add_argument('--data-path', '-d', type=str, default=DEFAULT_DATA_PATH,
                       help='Path where to save the imported data (default: imported_events_data)')
    parser.add_argument('--max-documents', '-m', type=int, default=DEFAULT_MAX_DOCUMENTS,
                       help=f'Maximum number of documents to fetch (default: {DEFAULT_MAX_DOCUMENTS})')
    args = parser.parse_args()
    
    docs = main(data_path=args.data_path, max_documents=args.max_documents)
