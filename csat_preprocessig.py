import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
import pickle

class csat_preprocessing:
    def __init__(self,csat:pd.DataFrame):
        csat.set_index("Unique id",inplace=True)
        columns=['channel_name', 'category', 'Sub-category', 'Customer Remarks',
       'Order_id', 'order_date_time', 'Issue_reported at', 'issue_responded',
       'Survey_response_Date', 'Customer_City', 'Product_category',
       'Item_price', 'connected_handling_time', 'Agent_name', 'Supervisor',
       'Manager', 'Tenure Bucket', 'Agent Shift', 'CSAT Score']
        #csat.replace("NaN",np.nan, inplace=True)
        if list(csat.columns)!=columns:
            raise ValueError("Dataframe columns do not match the expected columns.")
        else:
            self.csat=csat

            knn_impute_columns=[
                'channel_name', 'category', 'sub-category',  'customer_city', 'product_category','item_price', 'connected_handling_time', 'agent_name', 'supervisor',
                    'manager', 'tenure_bucket', 'agent_shift','issue_resolution_time'
                    ]
            label_columns=[
                'channel_name', 'category', 'sub-category',  'customer_city', 'product_category', 'agent_name', 'supervisor',
                'manager', 'tenure_bucket', 'agent_shift'
                    ]
            numeric_columns=['item_price', 'connected_handling_time', 'issue_resolution_time']

            self.knn_impute_columns=knn_impute_columns
            self.label_columns=label_columns
            self.numeric_columns=numeric_columns
        with open("knn_imputer.pkl","rb") as f:
            self.knn_imputer=pickle.load(f)
        with open("label_encoder.pkl","rb") as f:
            self.label_encoder=pickle.load(f)
        with open("cat_imputer.pkl","rb") as f:
            self.cat_imputer=pickle.load(f)



            pass
            

    

    def preprocessed(self):
        self.csat.drop_duplicates(inplace=True)
        self.csat.fillna({"Customer Remarks":"No_Remarks"}, inplace=True)
        # Sentiment score Function
        def sentiment_score(text):
            analyzer = SentimentIntensityAnalyzer()
            score = analyzer.polarity_scores(text)
            return score["compound"]
        
        st.write("Calculating sentiment scores for customer remarks...")
        st.warning("This step might take a while depending on the number of records in the dataset. Estimated time: 1 minutes per 10000 records.")
        self.csat["Customer Remarks"].apply(lambda x: np.nan if x=="No_Remarks" else sentiment_score(x))
        # Correcting the column names
        self.csat.columns=self.csat.columns.str.lower().str.replace(" ","_")
        # Calculating time Difference between issue reported and issue responded in minutes
        self.csat["issue_resolution_time"]=pd.to_datetime(self.csat["issue_responded"], format="%d/%m/%Y %H:%M")-pd.to_datetime(self.csat["issue_reported_at"], format="%d/%m/%Y %H:%M")
        self.csat["issue_resolution_time"]=self.csat["issue_resolution_time"].astype("timedelta64[s]").dt.seconds/60

        st.write("Imputing missing values using KNN Imputer...")
        st.warning("This step might take a while depending on the number of records in the dataset. Estimated time: 1 minutes per 10000 records.")
        self.csat[self.label_columns]=self.cat_imputer.transform(self.csat[self.label_columns])
        self.csat.fillna({"item_price":self.csat["item_price"].median()}, inplace=True)
        self.csat[self.label_columns]=self.label_encoder.transform(self.csat[self.label_columns])
        self.csat[self.knn_impute_columns]=self.knn_imputer.transform(self.csat[self.knn_impute_columns])
        self.csat["survey_response_time"]=pd.DataFrame(pd.to_datetime(self.csat["survey_response_date"], format="mixed").dt.date-pd.to_datetime(self.csat["issue_reported_at"], format="%d/%m/%Y %H:%M").dt.date)
        self.csat["survey_response_time"]=self.csat["survey_response_time"]/3600

        return self.csat