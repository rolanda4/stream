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

st.info('This is app predicts the likelihood of loan default given certain parameters!')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://github.com/rolanda4/stream/blob/main/cleaned_dataset.csv')
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

