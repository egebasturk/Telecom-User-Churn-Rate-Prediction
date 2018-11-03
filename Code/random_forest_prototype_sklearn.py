import numpy as np
import pandas as pa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Read data, separate features and the labels then split to train and test data
data = pa.read_csv("../Data/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn_Processed.csv")

features = data.iloc[:, 2:21].copy()
features = np.squeeze(np.asarray(features))

label_Column = data.iloc[:, 21].copy()
# Using Sklearn data split function
features_train, features_test, label_train, label__test = train_test_split(features, label_Column, test_size=0.40)

# Feature scaling. Needed since features vary currently a lot
scaler = StandardScaler()
scaler.fit(features_train)

features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

# Training
# TODO: given parameters randomly, learn the details
classifier = RandomForestClassifier(n_estimators=10, random_state=0)
classifier.fit(features_train, label_train)

# Prediction. Predict test data
predictions = classifier.predict(features_test)

# Evaluations
print(confusion_matrix(label__test, predictions))
print(classification_report(label__test, predictions))