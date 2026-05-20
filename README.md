# Customer Churn Prediction System

An end-to-end machine learning system that predicts customer churn probability using Logistic Regression, served via a production-ready REST API deployed on AWS Elastic Beanstalk.

## 🔗 Live API

http://churn-serving-env.eba-cfuqxid3.eu-north-1.elasticbeanstalk.com/predict

## 📊 Model Performance

- **Algorithm:** Logistic Regression (C=1.0)
- **AUC-ROC:** 0.84
- **Accuracy:** 80%+
- **Dataset:** 7,000+ customer records
- **Validation:** K-Fold Cross-Validation

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **ML:** Scikit-learn, NumPy, Pandas
- **API:** Flask, Gunicorn
- **Containerization:** Docker
- **Deployment:** AWS Elastic Beanstalk
- **Dependency Management:** Pipenv

## 📁 Project Structure

churn-prediction-project/
│
├── src/
│   └── predict.py        # Flask REST API
├── notebooks/            # EDA and training notebooks
├── data/                 # Dataset
├── model_C=1.0.bin       # Serialized model + DictVectorizer
├── train.py              # Model training script
├── Dockerfile
├── Pipfile
└── README.md

## 🚀 API Usage

### Predict Churn

**Endpoint:** `POST /predict`

**Request:**

```bash
curl -X POST http://churn-serving-env.eba-cfuqxid3.eu-north-1.elasticbeanstalk.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "contract": "two_year",
    "tenure": 12,
    "monthlycharges": 19.9
  }'
```

**Response:**
```json
{
  "churn": false,
  "churn_probability": 0.166
}
```

## 🏃 Run Locally

**With Docker:**

```bash
docker build -t churn-predictor .
docker run -p 9696:9696 churn-predictor
```

**Without Docker:**

```bash
pipenv install
pipenv run gunicorn --bind=0.0.0.0:9696 predict:app
```

Then call:
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"contract": "two_year", "tenure": 12, "monthlycharges": 19.9}'
```

## 🔧 Train the Model Yourself

```bash
pipenv run python train.py
```
This regenerates `model_C=1.0.bin` from the dataset.

## 📈 Key ML Decisions

- **Feature Engineering:** Mutual Information scoring to rank categorical variable importance
- **Encoding:** One-Hot Encoding for categorical features via DictVectorizer
- **Regularization:** Tuned C parameter to balance bias-variance tradeoff
- **Threshold:** 0.5 classification threshold (adjustable based on business need)

## 👤 Author
**Anamitra Ghosh**  
[GitHub](https://github.com/anamitraghosh03) | anamitraghosh07@gmail.com