# 💳 Credit Card Fraud Detection using Machine Learning

A Machine Learning-based web application that detects fraudulent credit card transactions using anomaly detection techniques and an XGBoost classifier. The project includes data preprocessing, model training, performance evaluation, and an interactive Streamlit dashboard for real-time predictions.


## 📌 Project Overview

Credit card fraud is one of the biggest challenges faced by financial institutions. This project uses machine learning algorithms to identify fraudulent transactions from highly imbalanced transaction data.

The application allows users to upload transaction data and instantly predicts whether each transaction is **Fraudulent** or **Legitimate**.


## 🚀 Features

- Data preprocessing and cleaning
- Handling imbalanced datasets using SMOTE
- Feature scaling with StandardScaler
- Anomaly Detection using:
  - Isolation Forest
  - Local Outlier Factor (LOF)
- Fraud Classification using XGBoost
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC Score
  - Confusion Matrix
- Streamlit web application
- Batch prediction using CSV upload

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Imbalanced-Learn (SMOTE)
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## 📂 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── creditcard.csv (Not included in GitHub)
│
├── models/
│   ├── xgboost_model.json
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── notebooks/
│   └── Fraud_Detection.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 📊 Dataset

This project uses the **Credit Card Fraud Detection Dataset** from Kaggle.

Dataset Link:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

> **Note:** The dataset is not included in this repository because GitHub limits files larger than 100 MB.

Download the dataset manually and place it inside the **data/** folder.


## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Supriya-2703/Credit-Card-Fraud-Detection.git
```

Move into the project folder

```bash
cd Credit-Card-Fraud-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```


## ▶️ Run the Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

The application will open at

```
http://localhost:8501
```


## 📈 Machine Learning Workflow

1. Load Dataset
2. Data Cleaning
3. Feature Scaling
4. Handle Class Imbalance using SMOTE
5. Train-Test Split
6. Train Isolation Forest
7. Train Local Outlier Factor
8. Train XGBoost Classifier
9. Evaluate Performance
10. Save Model
11. Deploy using Streamlit


## 📉 Model Evaluation

The project evaluates the model using:

- Confusion Matrix
- Classification Report
- ROC Curve
- ROC-AUC Score
- Precision
- Recall
- F1 Score


## 🖥️ Streamlit Dashboard

The web application allows users to:

- Upload transaction CSV files
- Predict fraudulent transactions
- View prediction results
- Download predicted output


## 📸 Screenshots

Add screenshots here after deployment.

Example:

```
screenshots/
    dashboard.png
    prediction.png
```


## 🔮 Future Improvements

- Real-time fraud detection using APIs
- Deep Learning Models
- Explainable AI (SHAP/LIME)
- Docker Deployment
- Cloud Deployment (AWS/Azure)
- User Authentication
- Database Integration


## 📄 License

This project is licensed under the MIT License.


## 👩‍💻 Author

**Supriya Bhade**

GitHub:
https://github.com/Supriya-2703

LinkedIn:
https://www.linkedin.com/in/supriya-bhade


## ⭐ If you like this project

Give this repository a ⭐ on GitHub!
