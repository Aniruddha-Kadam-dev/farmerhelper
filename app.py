from flask import Flask, render_template, request
from model.price_predictor import predict_prices_by_mandi

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    mandi_results = None
    if request.method == 'POST':
        crop = request.form['crop']
        date = request.form['date']
        state = request.form['state']
        district = request.form['district']
        mandi_results = predict_prices_by_mandi(crop, state, date)
        return render_template('result.html', crop=crop, date=date, state=state, district=district, mandi_results=mandi_results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) 