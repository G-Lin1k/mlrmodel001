from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from your Firebase site

# Load model
with open('mlr_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    df = pd.read_csv(file)

    # Adjust these feature columns based on your model
    X = df[['feature1', 'feature2', 'feature3']]
    predictions = model.predict(X)
    df['prediction'] = predictions
    return df.to_json(orient='records')

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
