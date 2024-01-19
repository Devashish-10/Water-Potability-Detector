import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score
# Load the dataset
dataset = pd.read_csv('water_potability.csv')
# Display the first few rows of the DataFrame
print(dataset.head())
# Handling missing values (if any)
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
# Scaling numeric features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed.drop('Potability', axis=1)), columns=df_imputed.columns[:-1])
# Concatenate the scaled features and the target variable
df_preprocessed = pd.concat([df_scaled, df_imputed['Potability']], axis=1)
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_preprocessed.drop('Potability', axis=1), df_preprocessed['Potability'], test_size=0.2, random_state=42)
# Display information about the preprocessed Dataset
print(df_preprocessed.info())
# Display statistics of the preprocessed Dataset
print(df_preprocessed.describe())
# Create an SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
# Train the model
svm_model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = svm_model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of SVM model: {accuracy * 100:.2f}%')
# Save the trained model 
model_filename = 'waterpotability_model.pkl'
joblib.dump(svm_model, model_filename)
