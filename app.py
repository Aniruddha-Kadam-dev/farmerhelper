from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from model.price_predictor import predict_prices_by_mandi, predict_price_ml
import json
import os
import pandas as pd
import io

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flash messages

# Helper to load metrics
METRICS_PATH = os.path.join('model', 'metrics.json')
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            return json.load(f)
    return None

# Helper to load dropdown options from last_fetch.csv
DATA_PATH = os.path.join('model', 'last_fetch.csv')
def load_options():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        crops = sorted(df['crop'].dropna().unique())
        states = sorted(df['state'].dropna().unique())
        districts = sorted(df['district'].dropna().unique())
        return crops, states, districts, df
    return [], [], [], pd.DataFrame()

@app.route('/districts')
def districts():
    state = request.args.get('state')
    _, _, _, df = load_options()
    if not state or df.empty:
        return jsonify([])
    districts = sorted(df[df['state'] == state]['district'].dropna().unique())
    return jsonify(districts)

@app.route('/', methods=['GET', 'POST'])
def index():
    mandi_results = None
    ml_prediction = None
    mandi_used = None
    metrics = load_metrics()
    crops, states, districts, _ = load_options()
    selected_crop = selected_state = selected_district = ''
    if request.method == 'POST':
        crop = request.form['crop']
        date = request.form['date']
        state = request.form['state']
        district = request.form['district']
        selected_crop = crop
        selected_state = state
        selected_district = district
        mandi_results = predict_prices_by_mandi(crop, state, date)
        if mandi_results:
            mandi_used = mandi_results[0]['mandi']
            ml_prediction = mandi_results[0]['price']
        return render_template('result.html', crop=crop, date=date, state=state, district=district, mandi_results=mandi_results, ml_prediction=ml_prediction, mandi_used=mandi_used, metrics=metrics)
    return render_template('index.html', crops=crops, states=states, districts=districts, selected_crop=selected_crop, selected_state=selected_state, selected_district=selected_district)

@app.route('/download')
def download_csv():
    crop = request.args.get('crop')
    state = request.args.get('state')
    date = request.args.get('date')
    mandi_results = predict_prices_by_mandi(crop, state, date)
    df = pd.DataFrame(mandi_results)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='mandi_predictions.csv')

@app.route('/newsletter', methods=['POST'])
def newsletter():
    email = request.form.get('email')
    # Here you would add logic to save the email or send to a service
    flash(f'Thank you for subscribing, {email}!', 'success')
    return redirect(url_for('index'))

@app.route('/data-availability')
def data_availability():
    crop = request.args.get('crop')
    state = request.args.get('state')
    district = request.args.get('district')
    DATA_PATH = os.path.join('model', 'last_fetch.csv')
    if not os.path.exists(DATA_PATH):
        return jsonify({'count': 0, 'can_predict': False})
    df = pd.read_csv(DATA_PATH)
    query = (df['crop'] == crop) & (df['state'] == state)
    if district:
        query = query & (df['district'] == district)
    count = df[query].shape[0]
    can_predict = count > 0
    return jsonify({'count': int(count), 'can_predict': can_predict})

if __name__ == '__main__':
    app.run(debug=True) 