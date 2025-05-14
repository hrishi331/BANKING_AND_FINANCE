import dateutil.utils
import numpy as np
import seaborn as sns
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

# header
st.header("Predict next open")

# inputs 
to_date = st.date_input('Select predict date',value='today')

# Select date range
# to_date = datetime.date(year = 2025 , month = 5 , day = 9)
from_date = to_date - datetime.timedelta(days=4500)



# Load data
df = yf.download(tickers="^NSEI",
                 start=from_date,
                 end=to_date)

# Rename columns
df.columns = ['Close','High','Low','Open','Volume']

# Convert data stationary [convert to %change]

def make_stationary(df,i):
    df[i] = ((df[i] - df[i].shift(1))/df[i].shift(1))*100
    df = df.dropna()
    return df

for i in df.columns:
    df = make_stationary(df,i)

# Shift Open by -1 , NxtOpen
df['NxtOpen'] = df['Open'].shift(-1)


# Targte and feature segregation
X = df.drop('NxtOpen',axis=1)
y = df['NxtOpen']

# Select 2500 days data back fromk prediction date
X = X[-2501:]
y = y[-2501:]


# train test split
X_train = X[:2000]
X_test = X[2000:]
y_train = y[:2000]
y_test = y[2000:]

# Create features
def create_features(df):
    df['day'] = [i.day for i in df.index]
    df['month'] = [i.month for i in df.index]
    df['weekday'] = [(i.weekday()+1) for i in df.index]
    df['weekno'] = [i.isocalendar()[1] for i in df.index]
    df['quarter'] = [int(np.ceil(i/3)) for i in df['month']]

    for j in ['day','weekday','month','weekno','quarter']:
        for i in ['Close','High','Low','Open','Volume']:
            data = df.groupby(by=j)[i].std().to_dict()
            df[f'{i}_{j}_Std'] = df[i].replace(data)
    return df

X_train = create_features(X_train)
X_test = create_features(X_test)

# Infifinity address
def address_infinity(df):
    for i in df:
        df[i] = df[i].replace(np.inf,np.nan)
        df[i] = df[i].fillna(df[i].max())
    return df

X_train = address_infinity(X_train)
X_test = address_infinity(X_test)

# Sample,test resegregation
X_sample = X_test.iloc[-1:]

X_test = X_test.iloc[:500]
y_test = y_test.iloc[:500]

# # Modelling
reg = RandomForestRegressor(random_state=11,max_depth=10)
reg.fit(X_train,y_train)

y_pred = pd.Series(reg.predict(X_test),index=y_test.index)
error = [i-j for i,j in zip(y_test,y_pred)]

r2 = round(r2_score(y_test,y_pred),2)
rmse = round(root_mean_squared_error(y_test,y_pred),2)
sample_pred = round(reg.predict(X_sample)[0],2)
obs_number = f"(rows,columns) : {X.shape}"
train_data = f"(rows,columns) : {X_train.shape}"
test_data = f"(rows,columns) : {X_test.shape}"
prediction_data = f"(rows,columns) : {X_sample.shape}"


if st.button(label='Predict'):
    with st.container(border=True):
        st.header('Results')

        st.subheader('Next Open Prediction')
        st.write(f'Next open on {to_date} likely : {sample_pred} %')

        st.subheader('Data used')
        st.write(f"Total data : {obs_number}")
        st.write(f"train data : {train_data}")
        st.write(f"test data : {test_data}")
        st.write(f'Prediction_Xdata : {prediction_data} ')

        st.subheader('Performance Metrics')
        st.write(f'Root Mean Squared Error : {rmse}')
        st.write(f'R2 Score : {r2}')

        fig = plt.figure(figsize=(10,7))

        sns.lineplot(y_test[-50:],color='black')
        sns.lineplot(y_pred[-50:],color='orange')

        sns.scatterplot(y_test[-50:],color='black',label='actual')
        sns.scatterplot(y_pred[-50:],color='orange',label='predicted')
        # plt.vlines(x=[y_test[-50:].index],
        #         ymin=[min(i,j) for i,j in zip(y_test[-50:],y_pred[-50:])],
        #         ymax=[max(i,j) for i,j in zip(y_test[-50:],y_pred[-50:])])
        plt.xticks(rotation = 90)
        plt.title('Prediction for latest 50 instances back from prediction date')
        plt.legend()
        st.pyplot(fig)

        st.subheader('Error distribution of test data')
        fig1 = plt.figure(figsize=(10,7))
        sns.histplot(error,kde=True)
        st.pyplot(fig1)