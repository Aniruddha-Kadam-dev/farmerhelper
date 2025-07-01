import json
import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), 'crop_prices.json')

def load_data():
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def predict_prices_by_mandi(crop, state, date):
    df = load_data()
    # Filter for crop, state, and date
    match = df[(df['crop'].str.lower() == crop.lower()) &
               (df['state'].str.lower() == state.lower()) &
               (df['date'] == date)]
    if not match.empty:
        # Return mandi, district, and price for each matching mandi
        return match[['mandi', 'district', 'price']].to_dict(orient='records')
    # Fallback: all mandis for crop+state, with their latest available price
    fallback = df[(df['crop'].str.lower() == crop.lower()) &
                  (df['state'].str.lower() == state.lower())]
    if not fallback.empty:
        # Get latest price for each mandi
        fallback['date'] = pd.to_datetime(fallback['date'])
        idx = fallback.groupby('mandi')['date'].idxmax()
        latest = fallback.loc[idx]
        return latest[['mandi', 'district', 'price', 'date']].to_dict(orient='records')
    return []

def predict_price(crop, state, district, date):
    df = load_data()
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