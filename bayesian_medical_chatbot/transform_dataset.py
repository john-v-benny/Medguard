"""
Transform the Kaggle dataset from symptom-list format to binary format.

Input format: Disease, Symptom_1, Symptom_2, ..., Symptom_17
Output format: Disease, symptom1, symptom2, ..., symptomN (all binary 0/1)
"""

import pandas as pd
import numpy as np

def transform_dataset(input_file, output_file):
    """Transform symptom-list format to binary format."""
    
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original shape: {df.shape}")
    print(f"Diseases: {df['Disease'].nunique()}")
    
    # Get all unique symptoms across all symptom columns
    symptom_columns = [col for col in df.columns if col.startswith('Symptom_')]
    all_symptoms = set()
    
    for col in symptom_columns:
        symptoms = df[col].dropna().str.strip()
        all_symptoms.update(symptoms)
    
    all_symptoms = sorted(list(all_symptoms))
    print(f"Unique symptoms found: {len(all_symptoms)}")
    
    # Create binary dataframe
    binary_data = []
    
    for idx, row in df.iterrows():
        # Get disease
        disease = row['Disease']
        
        # Get all symptoms for this row
        row_symptoms = []
        for col in symptom_columns:
            symptom = row[col]
            if pd.notna(symptom):
                row_symptoms.append(symptom.strip())
        
        # Create binary vector
        binary_row = {'prognosis': disease}
        for symptom in all_symptoms:
            binary_row[symptom] = 1 if symptom in row_symptoms else 0
        
        binary_data.append(binary_row)
    
    # Create dataframe
    binary_df = pd.DataFrame(binary_data)
    
    # Reorder columns: prognosis first, then all symptoms alphabetically
    cols = ['prognosis'] + sorted([col for col in binary_df.columns if col != 'prognosis'])
    binary_df = binary_df[cols]
    
    # Save
    binary_df.to_csv(output_file, index=False)
    
    print(f"\nTransformed shape: {binary_df.shape}")
    print(f"Saved to: {output_file}")
    print(f"\nSample row:")
    print(binary_df.iloc[0])
    
    return binary_df

if __name__ == "__main__":
    # Transform both training and testing files
    print("=" * 80)
    print("TRANSFORMING TRAINING DATA")
    print("=" * 80)
    train_df = transform_dataset('data/Training.csv', 'data/Training_binary.csv')
    
    print("\n" + "=" * 80)
    print("TRANSFORMING TESTING DATA")
    print("=" * 80)
    test_df = transform_dataset('data/Testing.csv', 'data/Testing_binary.csv')
    
    print("\n" + "=" * 80)
    print("TRANSFORMATION COMPLETE!")
    print("=" * 80)
    print("\nNew files created:")
    print("  - data/Training_binary.csv")
    print("  - data/Testing_binary.csv")
    print("\nUpdate your notebook to use these files instead!")
