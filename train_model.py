
# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
data = pd.read_csv(r"C:\Users\ASUS\Downloads\adult 3.csv")


# Clean column names
data.columns = data.columns.str.strip()

# Drop missing values
data.dropna(inplace=True)

# Categorical columns to encode
categorical_cols = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'gender', 'native-country'
]

# Initialize label encoder
encoder = LabelEncoder()

# Encode all categorical columns
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

# Encode the target column
data['income'] = encoder.fit_transform(data['income'])

# Split into features and target
X = data.drop('income', axis=1)
y = data['income']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate and print accuracy
accuracy = model.score(X_test, y_test)
print("✅ Model trained successfully!")
print("📊 Test Accuracy:", round(accuracy * 100, 2), "%")

# Save the trained model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("🎉 Model saved as 'model.pkl'")
