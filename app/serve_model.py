from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load your saved model
MODEL_FILENAME = "fraud_detector_xgb.pkl"
print(f"Loading model from {MODEL_FILENAME} ...")
model = joblib.load(MODEL_FILENAME)
print(f"Model loaded: {type(model).__name__}")

app = Flask(__name__)

# API endpoint to predict fraud
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Expecting all features in same order as during training
        features = [
            data["V1"], data["V2"], data["V3"], data["V4"], data["V5"], data["V6"], data["V7"], data["V8"], data["V9"], data["V10"],
            data["V11"], data["V12"], data["V13"], data["V14"], data["V15"], data["V16"], data["V17"], data["V18"], data["V19"], data["V20"],
            data["V21"], data["V22"], data["V23"], data["V24"], data["V25"], data["V26"], data["V27"], data["V28"], data["LOG_AMOUNT"], data["HOUR"]
        ]

        prediction = model.predict([features])[0]
        return jsonify({"prediction": int(prediction), "fraud": bool(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
