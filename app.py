import json
import flask
import pickle
from custom_objects import DenseTransformer
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

file_name = "models/model.pkl"
with open(file_name, 'rb') as f:
    model = pickle.load(f)

@app.route('/main.html', methods=['GET', 'POST'])
def predict():
    news_title = request.form['news_title']
    pct_sport = model.predict_proba([news_title])[0][1]
    if pct_sport >= 0.5:
        response = '%.1f%% a sport news' % (100 * pct_sport)
    else:
        response = '%.1f%% not a sport news' % (100 * (1 - pct_sport))
    return {'result': response}

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('main.html')
    elif request.method == 'POST':
        predictions = predict()
        return render_template('main.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)