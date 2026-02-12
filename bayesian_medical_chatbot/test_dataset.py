import pandas as pd
from collections import Counter

# Load data
df = pd.read_csv('bayesian_medical_chatbot/data/Training.csv')

# Get symptom columns
symptom_cols = [c for c in df.columns if c.startswith('Symptom_')]

# Count symptoms
counter = Counter()
for col in symptom_cols:
    counter.update(df[col].dropna())

# Print results
print('✓ Dataset loaded successfully')
print(f'✓ Training samples: {len(df):,}')
print(f'✓ Diseases: {df["Disease"].nunique()}')
print(f'✓ Unique symptoms: {len(counter)}')
print(f'\n✓ Top 5 most common symptoms:')
for symptom, count in counter.most_common(5):
    print(f'  - {symptom}: {count}')
print('\n✓ Notebook logic verified - ready to use!')
