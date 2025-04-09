import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

st.header("Bank Customer Churn Predictor")
st.subheader("User Inputs")

CreditScore = st.number_input("Credit Score",min_value=200,max_value=1000)
Geography = st.selectbox("Region",['Germany','France','Spain'],index=0)
Gender = st.selectbox("Gender",['Male','Female'],index=0)
Age = st.number_input("Age",min_value=18,max_value=90)
Tenure = st.number_input("Tenure",min_value=0,max_value=10)
Balance = st.number_input("Balance",min_value=0,step=500)
NumOfProducts  = st.selectbox("Number Of Products",[1,2,3,4])
HasCrCard  = st.selectbox("Have credit card? [1 for 'yes' and 0 for 'no']",[0,1])
IsActiveMember  = st.selectbox("Is active member? [1 for 'yes' and 0 for 'no']",[0,1])
EstimatedSalary  = st.number_input("Estimated salary",min_value=5000,step=1000)
Complain  = st.selectbox("Complain? [1 for 'yes' and 0 for 'no']",[0,1])
Sat_Score  = st.selectbox("Satoisfaction Score",[1,2,3,4,5])
Card_Type = st.selectbox("Card type",['GOLD','SILVER','DIAMOND','PLATINUM'],index=0)
point_earned = st.number_input("points earned",min_value=100,step=1000)

X = {'CreditScore':CreditScore, 
'Geography':Geography, 
'Gender':Gender, 
'Age':Age, 
'Tenure':Tenure, 
'Balance':Balance,
'NumOfProducts':NumOfProducts, 
'HasCrCard':HasCrCard, 
'IsActiveMember':IsActiveMember, 
'EstimatedSalary':EstimatedSalary,
'Complain':Complain, 
'Satisfaction Score':Sat_Score, 
'Card Type':Card_Type, 
'Point Earned':point_earned}

st.subheader("Input Summary")
X = pd.DataFrame([X])
st.write(X)

# import pickles
with open("encoder_dict.pkl","rb") as file:
    encoder_dict = pickle.load(file)

with open("scaler.pkl","rb") as file:
    scaler = pickle.load(file)

with open("model.pkl","rb") as file:
    model = pickle.load(file)

# Process data
for i in encoder_dict:
    X[i] = encoder_dict[i].transform(X[i])

X = pd.DataFrame(scaler.transform(X),columns=X.columns)

# Predicting 
y_pred = model.predict(X)


if st.button("Submit"):

    bar = st.progress(0)

    for i in range(101):
        time.sleep(0.05)
        bar.progress(i)

    bar.success("Task completed")

    st.subheader("Result")
    if y_pred==1:
        st.write("**Customer will churn!**")
    else:
        st.write("**Customer will not churn!**")