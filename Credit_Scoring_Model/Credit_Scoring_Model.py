
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



data = pd.read_csv("credit_score.csv")

print("Dataset Shape:", data.shape)
print(data.head())



if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)



data = data.dropna()



label_encoder = LabelEncoder()

categorical_columns = data.select_dtypes(include=['object']).columns

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])



X = data.drop("Credit_Score", axis=1)
y = data["Credit_Score"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)



y_pred = model.predict(X_test)




print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))