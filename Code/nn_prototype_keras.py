import os
# Backend change
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pa

data = pa.read_csv("../Data/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn_Processed.csv")
#print(data)
features = data.iloc[:, 2:21].copy()
features = np.squeeze(np.asarray(features))
#print(features)
target_Column = data.iloc[:, 21].copy()
#print(features) #-DEBUG

# Model creation
model = Sequential()
model.add(Dense(30, input_dim=19, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# Model compiliation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model fitting
model.fit(features, target_Column, epochs=100, batch_size=5)

# Evaluate the model
#scores = model.evaluate(features, target_Column)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Predict
predictions = model.predict(features)
# Round predictions
predictions = [round(x[0]) for x in predictions]
mat ={'Real': np.squeeze(np.asarray(target_Column)), 'Predicted': np.squeeze(np.asarray(predictions))}
dataFrame = pa.DataFrame(data=mat)
dataFrame.to_csv("../Data/telco-customer-churn/Predictions.csv")
