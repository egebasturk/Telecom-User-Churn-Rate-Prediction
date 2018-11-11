#import os
# Backend change
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS'] = 'device=cuda0, floatX=float32'

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pa
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# This reads data from the csv and save it to a data frame object
data = pa.read_csv("../Data/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn_Processed.csv")
#print(data)
# following separate features and the labels
features = data.iloc[:, 2:21].copy()
features = np.squeeze(np.asarray(features))
#print(features)
label_Column = data.iloc[:, 21].copy()

# Using Sklearn data split function
features_train, features_test, label_train, label__test = train_test_split(features, label_Column, test_size=0.40)

# Feature scaling. Needed since features vary currently a lot
scaler = StandardScaler()
scaler.fit(features_train)

features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

# Model creation
# tells keras to create the model
model = Sequential()
model.add(Dense(20, input_dim=19, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# Model compiliation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model fitting
model.fit(features_train, label_train, epochs=100, batch_size=5)

# Evaluate the model
scores = model.evaluate(features_test, label__test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Predict
predictions = model.predict(features_test)
# Round predictions
predictions = [round(x[0]) for x in predictions]

# Saves prediction to a new csv for manual control
mat ={'Real': np.squeeze(np.asarray(label__test)), 'Predicted': np.squeeze(np.asarray(predictions))}
dataFrame = pa.DataFrame(data=mat)
dataFrame.to_csv("../Data/telco-customer-churn/Predictions.csv")

print(confusion_matrix(label__test, predictions))
print(classification_report(label__test, predictions))
print("Accuracy:" + str(accuracy_score(label__test, predictions, normalize=True, sample_weight=None) * 100) + "%")