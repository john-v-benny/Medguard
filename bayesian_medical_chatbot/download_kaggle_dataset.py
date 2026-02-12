"""
Download the Disease Symptom Prediction dataset from Kaggle.

This script downloads the original Kaggle dataset:
https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset

Requirements:
- kaggle package installed (pip install kaggle)
- Kaggle API credentials configured (~/.kaggle/kaggle.json)
"""

import os
import zipfile
from pathlib import Path

def download_dataset():
    """Download and extract the Kaggle dataset."""
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("Downloading dataset from Kaggle...")
    print("Dataset: disease-symptom-description-dataset")
    
    try:
        # Download using Kaggle API
        os.system("kaggle datasets download -d itachi9604/disease-symptom-description-dataset -p data")
        
        # Extract the zip file
        zip_path = data_dir / "disease-symptom-description-dataset.zip"
        
        if zip_path.exists():
            print(f"\nExtracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Remove the zip file
            zip_path.unlink()
            print("✓ Extraction complete!")
            
            # List downloaded files
            print("\nDownloaded files:")
            for file in sorted(data_dir.glob("*.csv")):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")
            
            print("\n✓ Dataset download complete!")
            
        else:
            print("❌ Error: Zip file not found. Please check:")
            print("  1. Kaggle API is installed: pip install kaggle")
            print("  2. Kaggle credentials are configured in ~/.kaggle/kaggle.json")
            print("  3. You have accepted the dataset terms on Kaggle website")
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset")
        print("2. Click 'Download' button")
        print("3. Extract the files to the 'data/' directory")

if __name__ == "__main__":
    download_dataset()
