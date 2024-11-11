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

 import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/rolanda4/stream/refs/heads/main/cleaned_dataset.csv')

# Define features and target
X_raw = df.drop('Default', axis=1)
y_raw = df.Default.map({'N': 0, 'Y': 1})

# Sidebar for user inputs
with st.sidebar:
    st.header('Input features')
    Age = st.slider('Age (yrs)', 20, 78, 27)
    Income = st.slider('Income (Ghc)', 4000, 2039784, 65878)
    Home = st.selectbox('Home', ('OWN', 'MORTGAGE', 'RENT', 'OTHER'))
    Emp_length = st.slider('Emp_length (yrs)', 0, 41, 5)
    Intent = st.selectbox('Intent', ('EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'))
    Amount = st.slider('Amount (mm)', 500.00, 35000.00, 9588.19)
    Rate = st.slider('Rate', 5.42, 23.22, 11.00)
    Cred_length = st.slider('Cred_length (months)', 2.00, 30.00, 5.80)
    
    # Add a button to run the prediction
    if st.button("Run Prediction"):
        
        # Convert user inputs into a DataFrame
        input_data = pd.DataFrame({
            'Age': [Age],
            'Income': [Income],
            'Home': [Home],
            'Emp_length': [Emp_length],
            'Intent': [Intent],
            'Amount': [Amount],
            'Rate': [Rate],
            'Cred_length': [Cred_length]
        })

        # Combine input data with original dataset for consistent encoding
        input_parameters = pd.concat([input_data, X_raw], axis=0)
        df_encoded = pd.get_dummies(input_parameters, columns=['Home', 'Intent'])

        # Split the encoded data back into user input and training data
        input_row = df_encoded.iloc[0:1]
        X_encoded = df_encoded.iloc[1:]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_raw, test_size=0.2, random_state=42)

        # Apply StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        input_row_scaled = scaler.transform(input_row)

        # Model training and prediction
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Predict on user input
        prediction = model.predict(input_row_scaled)
        prediction_proba = model.predict_proba(input_row_scaled)

        # Display results
        st.write("Prediction:", "Default" if prediction[0] == 1 else "No Default")
        st.write("Prediction Probability:", prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0])
