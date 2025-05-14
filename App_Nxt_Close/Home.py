import streamlit as st

st.set_page_config(page_title='Home',initial_sidebar_state='expanded')
st.header("Next Close Prediction")
st.subheader("Read Me First")
st.write("""
         1.This application is used for prediction of next close.\n
         2.Next close prediction is in % change terms.\n 
         i.e. % change in close is w.r.t. previous close.\n 
         3.This application is used for prediction of next close on historical date or current day at latest.\n
         4.This application can not be used to predict close after 2,3,.....days later.\n
         5.Application provides data shapes of train,test and prediction data.\n
         6.It provides prediction of past 50 instances from prediction dates along with actual observations for comparison.\n
         7.This is prediction model hence performnace metrics is bound to change with prediction date.\n
         8.To predict close on current day this application can be used just after market opens i.e. just after 9:15 a.m.  
""")
if st.button("Application >>"):
    st.switch_page("pages/1_App.py")
