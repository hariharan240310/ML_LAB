# Parking Availability Prediction
# Algorithm: Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("parking_birmingham.csv")

print("Parking Availability Dataset:")
print(data.head())

# Features and target
X = data[["Day", "Time", "Capacity", "Occupied_Spaces"]]
y = data["Status"]

# Encode categorical columns
X = pd.get_dummies(X, columns=["Day", "Time"], dtype=int)

# Encode target labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict parking availability
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=encoder.inverse_transform([0, 1]),
        zero_division=0
    )
)

# Predict a new parking status
new_parking = pd.DataFrame([{
    "Day": "Monday",
    "Time": "Morning",
    "Capacity": 100,
    "Occupied_Spaces": 40
}])

new_parking = pd.get_dummies(
    new_parking, columns=["Day", "Time"], dtype=int
)

new_parking = new_parking.reindex(
    columns=X.columns, fill_value=0
)

prediction = model.predict(new_parking)
status = encoder.inverse_transform(prediction)[0]

print("\nPredicted Parking Status:", status)