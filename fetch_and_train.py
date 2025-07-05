import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import os
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import numpy as np

# User: Add your API key here
API_KEY = '579b464db66ec23bdd000001c5a31678fab4415f47f01852437363c0'
API_URL = 'https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24'

params = {
    'api-key': API_KEY,
    'format': 'json',
    'limit': 1000  # You can increase this or loop for more data
}

# Example: To filter by state, commodity, or date range, add to params:
# params['state'] = 'Madhya Pradesh'
# params['commodity'] = 'Wheat'
# params['from'] = '01/01/2020'
# params['to'] = '31/12/2020'

MAX_RECORDS = 50000  # Set your desired max
BATCH_SIZE = 1000
all_records = []

print('Fetching data from Agmarknet API...')
for offset in range(0, MAX_RECORDS, BATCH_SIZE):
    params['offset'] = offset
    params['limit'] = BATCH_SIZE
    response = requests.get(API_URL, params=params)
    data = response.json()
    batch_count = len(data['records']) if 'records' in data else 0
    print(f"Batch at offset {offset}: {batch_count} records fetched.")
    if 'records' not in data or not data['records']:
        break
    all_records.extend(data['records'])
    print(f"Total records fetched so far: {len(all_records)}")
    time.sleep(0.5)  # Be polite to the API

print(f"Total records fetched: {len(all_records)}")
df = pd.DataFrame(all_records)
print('Columns in raw DataFrame:', df.columns.tolist())
print('First row of raw DataFrame:')
print(df.head(1).to_dict(orient='records'))

# Drop rows with missing values in key columns (use original names)
df = df.dropna(subset=['Commodity', 'State', 'District', 'Market', 'Arrival_Date', 'Modal_Price'])

# Rename columns to standard names
df = df.rename(columns={
    'Commodity': 'crop',
    'State': 'state',
    'District': 'district',
    'Market': 'mandi',
    'Arrival_Date': 'date',
    'Modal_Price': 'price'
})

# Convert price to numeric
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])

# Feature engineering: extract year, month from date
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Select features and target
features = ['crop', 'state', 'district', 'mandi', 'year', 'month']
X = df[features]
y = df['price']

# Preprocessing for categorical features
categorical_features = ['crop', 'state', 'district', 'mandi']
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # year, month
)

# Build pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Training model...')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model training complete. Test R^2: {r2:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Save metrics to JSON
metrics = {'r2': r2, 'mae': mae, 'rmse': rmse}
with open('model/metrics.json', 'w') as f:
    json.dump(metrics, f)
print('Saved evaluation metrics to model/metrics.json')

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/price_model.joblib')
print('Model saved to model/price_model.joblib')

# Save the processed DataFrame for use in prediction
os.makedirs('model', exist_ok=True)
df.to_csv(os.path.join('model', 'last_fetch.csv'), index=False)
print('Saved last fetched data to model/last_fetch.csv') 