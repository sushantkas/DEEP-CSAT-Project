import pandas as pd


class csat_preprocessing:
    def __init__(self,csat:pd.DataFrame):
        csat.set_index("Unique id",inplace=True)
        columns=['channel_name', 'category', 'Sub-category', 'Customer Remarks',
       'Order_id', 'order_date_time', 'Issue_reported at', 'issue_responded',
       'Survey_response_Date', 'Customer_City', 'Product_category',
       'Item_price', 'connected_handling_time', 'Agent_name', 'Supervisor',
       'Manager', 'Tenure Bucket', 'Agent Shift', 'CSAT Score']
        if list(csat.columns)!=columns:
            raise ValueError("Dataframe columns do not match the expected columns.")
        else:
            self.csat=csat
            

    def preprocessed(self):
        self.csat.drop_duplicates(inplace=True)
