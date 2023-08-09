import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('credit_card_data.csv')

# Sidebar for user input
st.sidebar.header('User Input')
amount = st.sidebar.number_input("Transaction Amount", value=100.0)
# Add other input fields here

# Preprocessing (replace this with your actual preprocessing steps)
X = data.drop(columns=['Class'])
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Preprocess input data
input_data = np.array([[amount]])  # Adjust as needed
# Add preprocessing for other input fields

# Make prediction
prediction = model.predict(input_data)

# Display the Streamlit app
st.title('Credit Card Fraud Detection')
st.write('Enter transaction details:')
# Add input fields for other features
if st.button('Predict'):
    if prediction[0] == 0:
        st.write("Prediction: Legitimate Transaction")
    else:
        st.write("Prediction: Fraudulent Transaction")
    
    # Calculate accuracy on the test data
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy on Test Data: {test_accuracy:.2%}")
