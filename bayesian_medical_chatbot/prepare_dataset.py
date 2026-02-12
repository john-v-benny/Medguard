"""
Prepare the Kaggle dataset for training and testing.

The original Kaggle dataset comes as a single dataset.csv file.
This script splits it into Training.csv and Testing.csv files.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_dataset():
    """Split the dataset into training and testing sets."""
    
    print("Loading dataset.csv...")
    df = pd.read_csv('data/dataset.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")
    print(f"Diseases: {df['Disease'].nunique()}")
    
    # Split into training (90%) and testing (10%)
    train_df, test_df = train_test_split(
        df, 
        test_size=0.1, 
        random_state=42,
        stratify=df['Disease']  # Ensure balanced split across diseases
    )
    
    # Save to separate files
    train_df.to_csv('data/Training.csv', index=False)
    test_df.to_csv('data/Testing.csv', index=False)
    
    print(f"\nTraining set: {len(train_df)} samples")
    print(f"Testing set: {len(test_df)} samples")
    print("\nFiles created:")
    print("  - data/Training.csv")
    print("  - data/Testing.csv")
    print("\nDataset preparation complete!")

if __name__ == "__main__":
    prepare_dataset()
