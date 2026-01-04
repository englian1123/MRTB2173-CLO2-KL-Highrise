import os
import pandas as pd

# Get the directory where the script is located
script_dir = os.path.dirname(__file__) 

# Construct the path to the csv
file_path = os.path.join(script_dir, '..', 'data', 'KLHighRise.csv')

print(file_path)

# Load the data
df = pd.read_csv(file_path)

# # Inspect the first few rows and info
print(df.head())
print(df.info())

# --------------

print(df['UnitLevel'].unique())
print(df['Unit'].unique())
print(df['PropertyType'].unique())
print(df['Tenure'].unique())
print(df['Mukim'].unique())


# --------------
print(f"Unique SchemeNames: {df['SchemeName'].nunique()}")


# --------------
# Check weird UnitLevel values
print(df['UnitLevel'].value_counts().tail(20))



