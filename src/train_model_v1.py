import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Get the directory where the script is located
script_dir = os.path.dirname(__file__) 

# Construct the path to the csv
csv_file_path = os.path.join(script_dir, '..', 'data', 'KLHighRise.csv')

# Load the data
df = pd.read_csv(csv_file_path)

# Clean UnitLevel
def clean_unit_level(val):
    val = str(val).upper().strip()
    if val in ['G', 'UG']: return 0
    if val == 'LG': return -1
    if val == 'B': return -2
    if 'A' in val: val = val.replace('A', '') 
    if 'B' in val: val = val.replace('B', '')
    try:
        return float(val)
    except:
        return np.nan

df['UnitLevel_Cleaned'] = df['UnitLevel'].apply(clean_unit_level)
df['UnitLevel_Cleaned'] = df['UnitLevel_Cleaned'].fillna(df['UnitLevel_Cleaned'].median())

# Transaction Date
date_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

def parse_date(date_str):
    try:
        parts = date_str.split('-')
        month = date_map[parts[0]]
        year = int(f"20{parts[1]}")
        return year, month
    except:
        return 2022, 1

df['Year'], df['Month'] = zip(*df['TransactionDate'].apply(parse_date))

# Features and Target
features = ['Mukim', 'Tenure', 'ParcelArea', 'UnitLevel_Cleaned', 'Year', 'Month']
X = df[features]
y = df['TransactionPrice']

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
categorical_features = ['Mukim', 'Tenure']
numeric_features = ['ParcelArea', 'UnitLevel_Cleaned', 'Year', 'Month']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train
model_pipeline.fit(X_train, y_train)

# Evaluation
y_pred = model_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R2 Score: {r2}")
print(f"MAE: {mae}")

# Save the full model trained on all data
model_pipeline.fit(X, y)

# Construct the path to the save the joblib file
joblib_file_path = os.path.join(script_dir, '..', 'model', 'model_v1.pkl')

joblib.dump(model_pipeline, joblib_file_path)

print("Model saved as model_v1.pkl")