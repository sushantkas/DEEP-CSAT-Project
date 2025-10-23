import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import pickle
import importlib
import warnings
import csat_preprocessig
warnings.filterwarnings("ignore")
importlib.reload(csat_preprocessig)
from csat_preprocessig import csat_preprocessing




with open("lightgbm_final_model.pkl","rb") as f:
    model=pickle.load(f)

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="E-commerce CSAT Input", layout="wide")
# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📁 Upload Your Dataset")

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:

    # Read dataset
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, index_col="Unique id")
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("✅ Dataset Loaded Successfully!")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
else:
    st.info("Please upload a dataset file to continue.")


if st.button("🔄 Preprocess and Predict CSAT Scores"):
    
    with st.spinner("Processing Dataset....", show_time=True):

        cdata=csat_preprocessing(df).preprocessed()
        
        cdata.drop(["customer_remarks","order_id","order_date_time","issue_reported_at", "issue_responded","survey_response_date"], inplace=True, axis=1)
        try:
            
            cdata.drop("csat_score", axis=1, inplace=True)
        except:
            pass

        csat=model.predict(cdata)
        cdata["csat_score"]=csat
        st.dataframe(cdata)
        st.download_button("Download file...", data=cdata.to_csv(index=False).encode('utf-8'), file_name="predicted_data.csv", mime="csv")


