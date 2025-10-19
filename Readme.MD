# Deep-CSAT: Customer Satisfaction Prediction using Deep Learning

A deep learning-based approach to predict Customer Satisfaction (CSAT) scores for e-commerce customer support interactions.

## Project Overview

This project implements a machine learning solution to predict customer satisfaction scores based on various customer support interaction parameters. It utilizes natural language processing for sentiment analysis and deep learning for prediction.

## Features

- Sentiment analysis of customer remarks using VADER
- Automated data preprocessing pipeline
- Feature engineering for temporal data
- Deep learning model for CSAT prediction
- Comprehensive EDA with UBM (Univariate, Bivariate, Multivariate) analysis

## Project Structure

```
├── csat_preprocessig.py      # Preprocessing module
├── EDA.ipynb                # Exploratory Data Analysis notebook
├── ml_training.ipynb        # Model training notebook
├── label_encoder.pkl        # Saved label encoder model
├── knn_imputer.pkl         # Saved KNN imputer model
├── cat_imputer.pkl         # Saved categorical imputer model
```

## Data Features

### Input Features:
- **Channel Features**: channel_name, category, sub-category
- **Customer Information**: customer_city, product_category
- **Agent Information**: agent_name, supervisor, manager, tenure_bucket, agent_shift
- **Interaction Metrics**: 
  - connected_handling_time
  - issue_resolution_time
  - sentiment_score (derived from customer remarks)
  - item_price

### Target Variable:
- CSAT Score

## Data Preprocessing

1. **Missing Value Treatment**:
   - Customer Remarks: Filled with "No_Remarks"
   - Numeric columns: KNN Imputation (k=5)
   - Categorical columns: Most frequent value imputation

2. **Feature Engineering**:
   - Sentiment Score calculation using VADER
   - Issue resolution time computation
   - Survey response time calculation
   - Categorical variable encoding using OrdinalEncoder

3. **Data Quality Checks**:
   - VIF analysis for multicollinearity
   - Distribution analysis
   - Correlation studies

## Key Insights from EDA

1. **Correlation Patterns**:
   - Strong correlation between sentiment_score and CSAT
   - Resolution time impacts customer satisfaction

2. **Distribution Analysis**:
   - CSAT scores show imbalanced distribution
   - Sentiment scores follow normal distribution
   - Time-based features show right-skewed patterns

3. **Channel Performance**:
   - Different channels show varying CSAT patterns
   - Product categories influence satisfaction levels

## Implementation Details

### Preprocessing Pipeline (`csat_preprocessig.py`):
```python
class csat_preprocessing:
    def __init__(self, csat):
        # Initialize preprocessing with data validation
        
    def preprocessed(self):
        # Complete preprocessing pipeline including:
        # - Data cleaning
        # - Feature engineering
        # - Encoding
        # - Imputation
```

### Model Features

**Important Features Identified**:
1. Sentiment score
2. Issue resolution time
3. Connected handling time
4. Channel name
5. Product category
6. Item price

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- vaderSentiment
- seaborn
- matplotlib
- streamlit

## Usage

1. Data Preprocessing:
```python
from csat_preprocessig import csat_preprocessing
preprocessor = csat_preprocessing(data)
processed_data = preprocessor.preprocessed()
```

2. Run EDA:
```bash
jupyter notebook EDA.ipynb
```

3. Train Model:
```bash
jupyter notebook ml_training.ipynb
```

## Future Improvements

1. Implement advanced NLP techniques for sentiment analysis
2. Add real-time prediction capabilities
3. Develop API endpoints for model serving
4. Include more feature engineering techniques
5. Implement model monitoring and retraining pipeline

## Contributors
- Individual Project

## License
[Specify your license here]
