import streamlit as st
import streamlit.components.v1 as components


# Navigation
pg = st.navigation([
    st.Page("pages/page1.py", title="Individual Prediction"),
    st.Page("pages/page2.py", title="Batch Prediction"),
])
pg.run()
