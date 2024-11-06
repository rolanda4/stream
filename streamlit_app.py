import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

st.title('💹 Credit Risk Analysis App')

st.info('This app predicts the likelihood of loan default given certain parameters!')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/rolanda4/stream/refs/heads/main/cleaned_dataset.csv')
  df

  df['Home'].replace('OWN',1, inplace=True)
  df['Home'].replace('MORTGAGE',2, inplace=True)
  df['Home'].replace('RENT',3, inplace=True)
  df['Home'].replace('OTHER',4, inplace=True)

  df['Intent'].replace('EDUCATION',1, inplace=True)
  df['Intent'].replace('MEDICAL',2, inplace=True)
  df['Intent'].replace('VENTURE',3, inplace=True)
  df['Intent'].replace('PERSONAL',4, inplace=True)
  df['Intent'].replace('HOMEIMPROVEMENT',5, inplace=True)
  df['Intent'].replace('DEBTCONSOLIDATION',6, inplace=True)

  df['Default'].replace('Y',1, inplace=True)
  df['Default'].replace('N',0, inplace=True)

  st.write('**X**')
  X_raw = df.drop('Default', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.Default
  y_raw

  # Input features
with st.sidebar:
  st.header('Input features')
  Age = st.slider('Age (yrs)', 20, 27, 78)
  Income = st.slider('Income (Ghc)', 4000, 2039784, 65878)
  Home = st.selectbox('Home', ('1', '2', '3','4'))
  Emp_length = st.slider('Emp_length (yrs)', 0, 41, 5)
  Intent = st.selectbox('Intent', ('1', '2', '3','4', '5', '6'))
  Amount = st.slider('Amount (mm)', 500.00, 35000.00, 9588.19)
  Rate = st.slider('Rate', 5.42, 23.22, 11.00)
  Cred_length = st.slider('Cred_length (months)', 2.00, 30.00, 5.80)
  
  # Create a DataFrame for the input features
  data = {'Age': Age,
          'Income': Income,
          'Home': Home,
          'Emp_length': Emp_length,
          'Intent': Intent,
          'Amount': Amount,
          'Rate': Rate,
          'Cred_length': Cred_length}
  input_df = pd.DataFrame(data, index=[0])
  input_parameters = pd.concat([input_df, X_raw], axis=0)
  
  input_df



