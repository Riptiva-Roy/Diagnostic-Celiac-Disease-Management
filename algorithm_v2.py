# Importing all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib  # For saving models

# reading the dataset
file_path = r"C:\Users\jessi\OneDrive - Princeton University\Princeton Hacks\Diagnostic-Celiac-Disease-Management\PCOS Dataset.csv"
data = pd.read_csv(file_path)

# Cleaning data set (converting strings to numbers)
data["AMH(ng/mL)"] = pd.to_numeric(data["AMH(ng/mL)"], errors='coerce')
data["II    beta-HCG(mIU/mL)"] = pd.to_numeric(data["II    beta-HCG(mIU/mL)"], errors='coerce')

data['Marraige Status (Yrs)'] = data['Marraige Status (Yrs)'].fillna(data['Marraige Status (Yrs)'].median())
data['II    beta-HCG(mIU/mL)'] = data['II    beta-HCG(mIU/mL)'].fillna(data['II    beta-HCG(mIU/mL)'].median())
data['AMH(ng/mL)'] = data['AMH(ng/mL)'].fillna(data['AMH(ng/mL)'].median())
data['Fast food (Y/N)'] = data['Fast food (Y/N)'].fillna(data['Fast food (Y/N)'].mode()[0])

# Clearing up the extra space in the column names (optional)
data.columns = [col.strip() for col in data.columns]

# Identifying non-numeric columns
non_numeric_columns = data.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_columns)

# Converting non-numeric columns to numeric where possible
for col in non_numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Dropping rows with any remaining non-numeric values
data.dropna(inplace=True)

# Preparing data for model training
X = data.drop(["PCOS (Y/N)", "Sl. No", "Patient File No.", "Marraige Status (Yrs)", "Blood Group","II    beta-HCG(mIU/mL)","TSH (mIU/L)","Waist:Hip Ratio"], axis=1) # dropping out index from features too
y = data["PCOS (Y/N)"]

# Splitting the data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Fitting the RandomForestClassifier to the training set
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Making prediction and checking the test set
pred_rfc = rfc.predict(X_test)
accuracy = accuracy_score(y_test, pred_rfc)
print(f"Accuracy: {accuracy:.2f}")

# Cross-validation score
cv_scores = cross_val_score(rfc, X_train, y_train, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("Best parameters from GridSearchCV:", grid_search.best_params_)

# Retraining the model with the best parameters
rfc_best = grid_search.best_estimator_
rfc_best.fit(X_train, y_train)

# Prediction and accuracy with the best model
pred_rfc_best = rfc_best.predict(X_test)
accuracy_best = accuracy_score(y_test, pred_rfc_best)
print(f"Best model accuracy: {accuracy_best:.2f}")

# Confusion Matrix and Classification Report
cm = confusion_matrix(y_test, pred_rfc_best)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, pred_rfc_best))

# Feature Importance
feature_importances = rfc_best.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=X.columns, y=feature_importances)
plt.title("Feature Importance in Random Forest Model")
plt.xticks(rotation=90)
plt.show()

# Save the model
joblib.dump(rfc_best, 'pcos_random_forest_model.pkl')
print("Model saved successfully!")

# Example: Collecting user input for the features
print("Please enter the following details:")

user_input = []

# Loop over the columns of X
for column in X.columns:
    # Prompt the user to enter the value for each feature
    value = float(input(f"Enter the value for {column}: "))
    # Append the input value to the user_input list
    user_input.append(value)

# Convert the user input to a NumPy array and reshape it to be 2D
user_input_reshaped = np.array(user_input).reshape(1, -1)

# Convert user input to DataFrame with feature names
user_input_df = pd.DataFrame(user_input_reshaped, columns=X.columns)

# Assuming the scaler used during training
scaler = StandardScaler()  # Normally you'd load a fitted scaler

# For demonstration, let's assume no scaling (remove if you're scaling the input)
user_input_scaled = user_input_df  # Remove this if you're applying scaling

# Get the probability of PCOS
probabilities = rfc_best.predict_proba(user_input_scaled)

# Extract probability for PCOS (class 1)
probability_pcos = probabilities[0][1]  # Probability of PCOS (class 1)
probability_non_pcos = probabilities[0][0]  # Probability of non-PCOS (class 0)

# Output the result
print(f"Probability of PCOS: {probability_pcos * 100:.2f}%")
print(f"Probability of non-PCOS: {probability_non_pcos * 100:.2f}%")
