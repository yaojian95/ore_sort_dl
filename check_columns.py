import pickle
import pandas as pd
import yaml

# Load config to get pickle path
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

pickle_path = config['data']['pickle_path']
print(f"Loading {pickle_path}...")

with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

# data[1] is grade_df as per dataset.py
df = data[1]
print("Columns:", df.columns.tolist())
