#!/usr/bin/env python
"""
Download the KDD Cup 1999 dataset.
"""
import os
import urllib.request
import gzip
import shutil

def download_kdd_cup_data():
    """Download the KDD Cup 1999 dataset."""
    # URLs for the dataset
    kdd_cup_url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz"
    output_gz = "data/raw/kddcup.data.gz"
    output_file = "data/raw/kddcup.data"
    
    # Create directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    
    # Download the gzipped file
    print(f"Downloading {kdd_cup_url}...")
    urllib.request.urlretrieve(kdd_cup_url, output_gz)
    print(f"Downloaded to {output_gz}")
    
    # Extract the gzipped file
    print(f"Extracting {output_gz}...")
    with gzip.open(output_gz, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Extracted to {output_file}")
    
    print("Download complete!")

if __name__ == "__main__":
    download_kdd_cup_data()