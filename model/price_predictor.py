import os
import pandas as pd
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'price_model.joblib')

# Load the model and the training data columns
_model = None
_df = None

def load_model_and_data():
    global _model, _df
    if _model is None or _df is None:
        _model = joblib.load(MODEL_PATH)
        # Load the training data columns from the model pipeline
        # We'll use the same columns as in fetch_and_train.py
        # For mandis, we need to reconstruct the list from the training data
        # We'll load the data again from the last fetch (if available)
        # For now, we can use the model's feature names if available
        # But for mandis, let's require the user to keep the last fetch DataFrame as a CSV
        if os.path.exists(os.path.join(os.path.dirname(__file__), 'last_fetch.csv')):
            _df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'last_fetch.csv'))
        else:
            raise Exception('last_fetch.csv not found. Please save the last fetched DataFrame as last_fetch.csv in the model directory.')
    return _model, _df

def predict_prices_by_mandi(crop, state, date):
    model, df = load_model_and_data()
    # Filter for crop and state
    mandis = df[(df['crop'].str.lower() == crop.lower()) & (df['state'].str.lower() == state.lower())]['mandi'].unique()
    results = []
    for mandi in mandis:
        # Use the first district for this mandi
        district = df[(df['mandi'] == mandi)]['district'].iloc[0]
        # Extract year and month from date
        dt = pd.to_datetime(date)
        year = dt.year
        month = dt.month
        X = pd.DataFrame([{
            'crop': crop,
            'state': state,
            'district': district,
            'mandi': mandi,
            'year': year,
            'month': month
        }])
        price = model.predict(X)[0]
        results.append({'mandi': mandi, 'district': district, 'price': round(price, 2), 'date': date})
    return results

def predict_price(crop, state, district, date):
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'crop_prices.csv'))
    # Try exact match
    match = df[(df['crop'].str.lower() == crop.lower()) &
               (df['state'].str.lower() == state.lower()) &
               (df['district'].str.lower() == district.lower()) &
               (df['date'] == date)]
    if not match.empty:
        return float(match.iloc[0]['price'])
    # Fallback: average for crop+state+date
    fallback = df[(df['crop'].str.lower() == crop.lower()) &
                  (df['state'].str.lower() == state.lower()) &
                  (df['date'] == date)]
    if not fallback.empty:
        return float(fallback['price'].mean())
    # Fallback: average for crop+state
    fallback2 = df[(df['crop'].str.lower() == crop.lower()) &
                   (df['state'].str.lower() == state.lower())]
    if not fallback2.empty:
        return float(fallback2['price'].mean())
    return None

def predict_price_ml(crop, state, district, mandi, date):
    model, _ = load_model_and_data()
    dt = pd.to_datetime(date)
    year = dt.year
    month = dt.month
    X = pd.DataFrame([{
        'crop': crop,
        'state': state,
        'district': district,
        'mandi': mandi,
        'year': year,
        'month': month
    }])
    price = model.predict(X)[0]
    return price 