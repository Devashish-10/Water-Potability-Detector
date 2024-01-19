import joblib
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
# Load the trained SVM model
model_filename = 'waterpotability_model.pkl'
svm_model = joblib.load(model_filename)
# Get feature names (excluding the target variable)
feature_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
# Get user input for feature values
user_input = {}
for feature in feature_names:
    user_input[feature] = float(input(f'Enter value for {feature}: '))
# Convert user input to a DataFrame
user_df = pd.DataFrame([user_input])
# Scale the user input features using the same scaler used during training
scaler = StandardScaler()
user_scaled = pd.DataFrame(scaler.fit_transform(user_df), columns=user_df.columns)
# Make predictions on the user input
user_prediction = svm_model.predict(user_scaled)
prediction_label = 'Potable' if user_prediction[0] == 1 else 'Not Potable'
# Display the prediction
print(f'\nBased on the provided input, the water is predicted to be: {prediction_label}')