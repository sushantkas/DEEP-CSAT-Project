import streamlit as st
import time
import pandas as pd
import seaborn as sns
import csat_preprocessig
import warnings
import importlib
warnings.filterwarnings("ignore")
importlib.reload(csat_preprocessig)
from csat_preprocessig import csat_preprocessing
import pickle
import numpy as np




uniques_df=pd.read_csv("uniques.csv")
city=pd.read_csv("customer_city.csv")
city_list=city["Customer_City"].tolist()

with open("lightgbm_final_model.pkl","rb") as f:
    model=pickle.load(f)
# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="E-commerce CSAT Input", layout="wide")
st.title("🛍️ Enter Individual Customer Record...")
st.markdown("Please fill in all fields for one record:")


# ----------------------------
# Input Fields (Except CSAT Score)
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    channel_name = st.selectbox("Channel Name", ['Outcall', 'Inbound', 'Email'])
    category = st.selectbox("Category",['Product Queries', 'Order Related', 'Returns', 'Cancellation',
       'Shopzilla Related', 'Payments related', 'Refund Related',
       'Feedback', 'Offers & Cashback', 'Onboarding related', 'Others',
       'App/website'])
    
    sub_category = st.selectbox("Sub-category",['Life Insurance', 'Product Specific Information',
       'Installation/demo', 'Reverse Pickup Enquiry', 'Not Needed',
       'Fraudulent User', 'Exchange / Replacement', 'Missing',
       'General Enquiry', 'Return request', 'Delayed',
       'Service Centres Related', 'Payment related Queries',
       'Order status enquiry', 'Return cancellation', 'Unable to track', 
       'Seller Cancelled Order', 'Wrong', 'Invoice request',
       'Priority delivery', 'Refund Related Issues', 'Signup Issues',
       'Online Payment Issues', 'Technician Visit',
       'UnProfessional Behaviour', 'Damaged', 'Product related Issues',
       'Refund Enquiry', 'Customer Requested Modifications',
       'Instant discount', 'Card/EMI', 'Shopzila Premium Related',
       'Account updation', 'COD Refund Details', 'Seller onboarding',
       'Order Verification', 'Other Cashback', 'Call disconnected',
       'Wallet related', 'PayLater related', 'Call back request',
       'Other Account Related Issues', 'App/website Related',
       'Affiliate Offers', 'Issues with Shopzilla App', 'Billing Related',
       'Warranty related', 'Others', 'e-Gift Voucher',
       'Shopzilla Rewards', 'Unable to Login', 'Non Order related',
       'Service Center - Service Denial', 'Payment pending',
       'Policy Related', 'Self-Help', 'Commission related'] )
    
    customer_remarks = st.text_area("Customer Remarks", placeholder="Enter remarks if any...")
    order_id = st.text_input("Order ID")
    order_date_time = st.date_input("Order Date Time (DD-MM-YYYY)", format="DD-MM-YYYY")
    issue_reported = st.date_input("Issue Reported At(DD-MM-YYYY)", format="DD-MM-YYYY")
    issue_responded = st.date_input("Issue Responded (Y/N) (DD-MM-YYYY)", format="DD-MM-YYYY")
    survey_response_date = st.date_input("Survey Response Date (DD-MM-YYYY)", format="DD-MM-YYYY")

with col2:
    customer_city = st.selectbox("Customer City", city_list, accept_new_options=False)
    product_category = st.selectbox("Product Category",['LifeStyle', 'Electronics', 'Mobile', 'Home Appliences',
       'Furniture', 'Home', 'Books & General merchandise', 'GiftCard',
       'Affiliates', None],
       accept_new_options=False)
    item_price = st.number_input("Item Price (₹)", min_value=0.0, value=100.0)
    connected_handling_time = st.number_input("Connected Handling Time (mins)", min_value=0.0, value=5.0)
    agent_name = st.selectbox("Agent Name", uniques_df["Agent_name"].tolist(), accept_new_options=False)
    supervisor = st.selectbox("Supervisor", ['Mason Gupta', 'Dylan Kim', 'Jackson Park', 'Olivia Wang',
       'Austin Johnson', 'Emma Park', 'Aiden Patel', 'Evelyn Kimura',
       'Nathan Patel', 'Amelia Tanaka', 'Harper Wong', 'Zoe Yamamoto',
       'Scarlett Chen', 'Sophia Sato', 'Wyatt Kim', 'Logan Lee',
       'Mia Patel', 'William Park', 'Emily Yamashita', 'Madison Kim',
       'Noah Patel', 'Oliver Nguyen', 'Elijah Yamaguchi',
       'Layla Taniguchi', 'Isabella Wong', 'Carter Park', 'Jacob Sato',
       'Ethan Tan', 'Mia Yamamoto', 'Brayden Wong', 'Ava Wong',
       'Landon Tanaka', 'Lucas Singh', 'Charlotte Suzuki',
       'Abigail Suzuki', 'Ethan Nakamura', 'Olivia Suzuki',
       'Alexander Tanaka', 'Lily Chen', 'Sophia Chen'])
    manager = st.selectbox("Manager", ['Jennifer Nguyen', 'Michael Lee', 'William Kim', 'John Smith',
       'Olivia Tan', 'Emily Chen'])
    tenure_bucket = st.selectbox("Tenure Bucket", ['On Job Training', '>90', '0-30', '31-60', '61-90'])
    agent_shift = st.selectbox("Agent Shift", ['Morning', 'Evening', 'Split', 'Afternoon', 'Night'])

# ----------------------------
# Submit Button
# ----------------------------
submit = st.button("🚀 Submit Record and Predict CSAT Score")

if submit:


    # Store record as dict
    record = {
        "channel_name": channel_name,
        "category": category,
        "Sub-category": sub_category,
        "Customer Remarks": customer_remarks,
        "Order_id": order_id,
        "order_date_time": order_date_time,
        "Issue_reported at": issue_reported,
        "issue_responded": issue_responded,
        "Survey_response_Date": survey_response_date,
        "Customer_City": customer_city,
        "Product_category": product_category,
        "Item_price": item_price,
        "connected_handling_time": connected_handling_time,
        "Agent_name": agent_name,
        "Supervisor": supervisor,
        "Manager": manager,
        "Tenure Bucket": tenure_bucket,
        "Agent Shift": agent_shift
    }

    record_df = pd.DataFrame([record])
    columns=['channel_name', 'category', 'Sub-category', 'Customer Remarks',
       'Order_id', 'order_date_time', 'Issue_reported at', 'issue_responded',
       'Survey_response_Date', 'Customer_City', 'Product_category',
       'Item_price', 'connected_handling_time', 'Agent_name', 'Supervisor',
       'Manager', 'Tenure Bucket', 'Agent Shift']
    record_df=record_df[columns]
    st.success("✅ Record Submitted Successfully!")
    st.dataframe(record_df)
    with st.spinner("🔄 Preprocessing the record..."):

        cdata=csat_preprocessing(record_df).preprocessed()
        cdata.drop(["customer_remarks","order_id","order_date_time","issue_reported_at", "issue_responded","survey_response_date"], inplace=True, axis=1)
        model_prediction=model.predict(cdata)
    st.markdown(f"### Predicted CSAT Score: **{int(model_prediction[0])}**")
    cdata["Predicted CSAT Score"]=model_prediction
    st.dataframe(cdata)
