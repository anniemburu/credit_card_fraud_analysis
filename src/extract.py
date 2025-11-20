import pandas as pd
from pathlib import Path
from zipfile import ZipFile
import os
from kaggle.api.kaggle_api_extended import KaggleApi



def extract_data():
    kaggle_dataset = 'mlg-ulb/creditcardfraud' 
    csv_file_name = "creditcard.csv"

    data_dir = Path('data')
    raw_data_dir = data_dir / 'raw'
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # Authenticate kaggle api
    api = KaggleApi()
    api.authenticate()

    temp_dir = data_dir / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {kaggle_dataset}...")

    # Use dataset_download_files for datasets (instead of competition_download_files)
    api.dataset_download_files(
        kaggle_dataset,
        path=str(temp_dir),
        unzip=False  # We'll handle extraction manually
    )

    zip_file_path = temp_dir / f"{kaggle_dataset.split('/')[-1]}.zip"

    with ZipFile(zip_file_path, 'r') as zf:
        print("\nExtracting files...")

        # Check if the expected file exists in the zip
        if csv_file_name not in zf.namelist():
            print(f"⚠️ {csv_file_name} not found in the zip file. Available files: {zf.namelist()}")
            
            # Try to find the actual CSV file name
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            if csv_files:
                csv_file_name = csv_files[0]
                print(f"Found alternative CSV file: {csv_file_name}")
            else:
                raise ValueError("No CSV files found in the dataset")
        
        # Extract and read the CSV file
        with zf.open(csv_file_name) as f:
            df = pd.read_csv(f)

        # Save as parquet
        output = raw_data_dir / f"creditcard_fraud_raw.parquet"
        df.to_parquet(output, index=False)


    #clean up
    os.remove(zip_file_path)
    os.rmdir(temp_dir)
    print("\n Clean Up Completed...")
    print("Data Extraction Completed.")
