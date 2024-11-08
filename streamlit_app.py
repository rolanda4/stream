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
  Home = st.selectbox('Home', ('OWN', 'MORTGAGE', 'RENT','OTHER'))
  Emp_length = st.slider('Emp_length (yrs)', 0, 41, 5)
  Intent = st.selectbox('Intent', ('EDUCATION', 'MEDICAL', 'VENTURE','PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'))
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

with st.expander('Input'):
  st.write('**Your Entries**')
  input_df



#encoding categorical values to become numerals
df_encoded = pd.get_dummies(input_parameters, columns=['Home', 'Intent','Default'])

#applying scalar to normalize
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_encoded)
df_scaled = pd.DataFrame(scaled_features, columns=df_encoded.columns)

# Encode X
X= df_encoded.drop('Default_N','Default_Y' axis=1)
X = df_encoded[1:]
input_row = df_encoded[:1]

# Encode y
y= np.argmax(df_encoded[['Default_N', 'Default_Y']].values, axis=1)
t
#target_mapper = {'N': 0,
                 'Y': 1,}
#def target_encode(val):
  #return target_mapper[val]

#y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X (input data)**')
  input_row
  st.write('**Encoded y**')
  y
# Model training and inference
## Train the ML model







