import streamlit as st

st.set_page_config(page_title="Home Page",initial_sidebar_state="expanded")

st.header("Home")

st.subheader("About Application")
st.write("This application is for bank and financial isntitutions. \
         This app predicts depending upoon user inputs if customer will churn or not.")

st.subheader("Application")
if st.button(label="Click for Application"):
    st.switch_page("pages/1_app.py")

st.subheader("Contact Us")
if st.button(label="Click to contact us."):
    st.switch_page("pages/2_Contact_Us.py")