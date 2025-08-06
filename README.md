# Credit Card Fraud Detection API 🚀

A Machine Learning project that detects fraudulent credit card transactions using an **XGBoost Classifier**.
The project includes **model training**, **Flask API deployment**, and **local API testing**.

---

## 📂 Project Structure

```
Fraud_Detection_Project/
│
├── notebooks/
│   └── Fraud_Detection.ipynb
│
├── model/
│   └── fraud_detector_xgb.pkl
│
├── app/
│   ├── serve_model.py         # Flask API
│   ├── requirements.txt       # Dependencies
│   └── test_request.py        # Local API test script
│
├── README.md
└── LICENSE
```

---

## 📊 Dataset

* Dataset: **Kaggle Credit Card Fraud Detection Dataset**
* 284,807 transactions, heavily imbalanced (\~0.17% fraud cases).
* Features: PCA-transformed variables (`V1`-`V28`), `Amount`, and `Time`.
* Target variable: `Class` (`1` = Fraud, `0` = Legitimate).

---

## 🧠 How I Trained the Model

1. **Data Preprocessing**

   * Loaded the dataset in a Jupyter Notebook.
   * Applied **log transformation** on `Amount` to normalize scale → `LOG_AMOUNT`.
   * Extracted **hour of transaction** from `Time` → `HOUR`.
   * Kept PCA features `V1`–`V28` and engineered features.

2. **Handling Imbalance**

   * Used **SMOTE** to oversample fraud cases in training data.

3. **Model Selection**

   * Chose **XGBoost Classifier** for:

     * High performance on imbalanced datasets.
     * Interpretability & feature importance.
   * Performed **hyperparameter tuning** via GridSearchCV.

4. **Model Evaluation**

   * Metrics: Precision, Recall, F1-score, ROC-AUC.
   * Achieved **high recall** to minimize missed frauds.

5. **Saving the Model**

   ```python
   import joblib
   joblib.dump(model, "fraud_detector_xgb.pkl")
   ```

📌 **Full training notebook**: [Fraud\_Detection.ipynb](notebooks/Fraud_Detection.ipynb)

---

## 🚀 API Deployment

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the API

```bash
python serve_model.py
```

This will start a Flask server at:

```
http://127.0.0.1:5000
```

---

## 📬 API Usage

**Endpoint:**
`POST /predict`

**Request Body (JSON):**

```json
{
    "V1": -1.2,
    "V2": 0.3,
    "V3": 0.1,
    "V4": -2.3,
    "V5": 0.0,
    "V6": 0.5,
    "V7": -0.4,
    "V8": 0.2,
    "V9": -1.1,
    "V10": 0.3,
    "V11": -0.2,
    "V12": -4.0,
    "V13": 0.0,
    "V14": 0.6,
    "V15": -0.8,
    "V16": 0.0,
    "V17": -0.3,
    "V18": 0.0,
    "V19": 0.1,
    "V20": 0.0,
    "V21": 0.0,
    "V22": -0.1,
    "V23": 0.0,
    "V24": 0.0,
    "V25": 0.0,
    "V26": -0.6,
    "V27": 0.0,
    "V28": 0.0,
    "LOG_AMOUNT": 6.2,
    "HOUR": 13
}
```

**Response:**

```json
{
    "prediction": 0,
    "fraud": false
}
```

---

## 🧪 Testing Locally

Run:

```bash
python test_request.py
```

---

## 📈 Results

* **High Recall** → fewer missed fraud cases.
* Balanced trade-off between **Precision** and **Recall**.

---

## 🔮 Future Improvements

* Deploy API to **Hugging Face Spaces** or **Render** for public use.
* Integrate real-time streaming with Kafka.
* Add model explainability with SHAP.

---

