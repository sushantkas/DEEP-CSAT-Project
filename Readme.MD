# Deep-CSAT: E-commerce Customer Satisfaction Prediction

A comprehensive machine learning solution for predicting customer satisfaction scores in e-commerce customer support interactions, featuring multiple ML models, deep learning approaches, and a Streamlit-based deployment interface.

## 🎯 Project Overview

This project implements an end-to-end machine learning solution to predict customer satisfaction (CSAT) scores based on various customer support interaction parameters. It utilizes advanced NLP for sentiment analysis, multiple machine learning models for prediction, and provides a user-friendly interface for real-time predictions.

## 🌟 Features

- **Advanced Data Processing**:
  - Sentiment analysis using VADER
  - Automated preprocessing pipeline
  - Feature engineering for temporal data
  - Robust handling of missing values

- **Multiple ML Models**:
  - LightGBM Classifier (Best performing)
  - Random Forest Classifier
  - Neural Network (ANN)
  - TabNet Implementation

- **MLflow Integration**:
  - Experiment tracking
  - Hyperparameter optimization
  - Model versioning
  - Performance metrics logging

- **Interactive Interface**:
  - Streamlit-based web application
  - Real-time predictions
  - User-friendly input forms
  - Visual result presentation

## 📁 Project Structure

```
├── csat_preprocessig.py        # Preprocessing module
├── ml_training.ipynb          # Model training and experimentation
├── EDA.ipynb                  # Exploratory Data Analysis
├── pages/
│   ├── page1.py              # Streamlit individual prediction page
│   └── page2.py              # Streamlit batch prediction page
├── models/
│   ├── lightgbm_final_model.pkl
│   ├── label_encoder.pkl
│   ├── knn_imputer.pkl
│   ├── cat_imputer.pkl
│   └── robust_scaler.pkl
└── data/
    └── eCommerce_Customer_support_data.csv
```

## 🔍 Machine Learning Pipeline

### 1. Data Preprocessing (`csat_preprocessig.py`)
- Custom preprocessing class implementation
- Automated handling of missing values
- Feature engineering pipeline
- Data validation and transformation

### 2. Model Development (`ml_training.ipynb`)
- Multiple model implementations:
  ```python
  # LightGBM (Best Performing Model)
  params = {
      "objective": "multiclass",
      "num_class": 5,
      "learning_rate": 0.211,
      "num_leaves": 118,
      "max_depth": 15,
      "is_unbalance": True
  }
  ```
- Neural Network Architecture:
  ```python
  model = Sequential([
      Dense(128, activation='relu'),
      Dropout(0.3),
      Dense(64, activation='relu'),
      Dense(5, activation='softmax')
  ])
  ```

### 3. Model Evaluation
- Metrics tracked:
  - Accuracy
  - F1 Score (Weighted)
  - Precision & Recall
  - Confusion Matrix
  - AUC-ROC

### 4. MLflow Integration
- Experiment tracking
- Hyperparameter tuning using Optuna
- Model versioning and artifact logging

## 📊 Key Features Used for Prediction

1. **Customer Interaction**:
   - Channel name
   - Category
   - Sub-category
   - Customer remarks (sentiment)

2. **Temporal Features**:
   - Issue resolution time
   - Connected handling time
   - Survey response time

3. **Agent Information**:
   - Agent name
   - Supervisor
   - Manager
   - Tenure bucket
   - Shift

4. **Transaction Details**:
   - Product category
   - Item price

## 🚀 Deployment

### Local Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run Streamlit app:
```bash
streamlit run app.py
```

### Usage
1. **Individual Predictions**:
   - Navigate to single prediction page
   - Fill in customer interaction details
   - Get instant CSAT prediction

2. **Batch Predictions**:
   - Upload CSV file with multiple records
   - Get predictions for all records
   - Download results

## 📈 Performance Metrics

- LightGBM Model:
  - Accuracy: 0.85
  - Weighted F1 Score: 0.84
  - AUC-ROC: 0.92

- Neural Network:
  - Accuracy: 0.83
  - Weighted F1 Score: 0.82
  - AUC-ROC: 0.90

## 🛠️ Requirements

- Python 3.8+
- Key Libraries:
  - pandas
  - numpy
  - scikit-learn
  - lightgbm
  - tensorflow
  - streamlit
  - mlflow
  - vaderSentiment

## 🔜 Future Improvements

1. Real-time model retraining pipeline
2. A/B testing framework for model comparison
3. Advanced feature engineering using deep learning
4. API deployment for microservice architecture
5. Model interpretability dashboard
6. Performance monitoring system

## 👥 Contributors
- Individual Project (Sushant Kashiram)

## 📄 License
MIT License

## 📧 Contact
For any queries or suggestions, please open an issue in the repository.
