#import os
# Backend change
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS'] = 'device=cuda0, floatX=float32'

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pa
import math

# This reads data from the csv and save it to a data frame object
data = pa.read_csv("../Data/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn_Processed.csv")
#print(data)
# following separate features and the labels
features = data.iloc[:, 2:21].copy()
features = np.squeeze(np.asarray(features))
#print(features)
label_Column = data.iloc[:, 21].copy()
#print(features) #-DEBUG

# Model creation
# tells keras to create the model
model = Sequential()
model.add(Dense(20, input_dim=19, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# Model compiliation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model fitting
data_div_ratio = 0.60 # 60% training 40% test
rows = 7043
training_rows = math.floor(rows * data_div_ratio)
model.fit(features[training_rows:], label_Column[training_rows:], epochs=100, batch_size=5)

# Evaluate the model
scores = model.evaluate(features[:training_rows], label_Column[:training_rows])
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Predict
predictions = model.predict(features)
# Round predictions
predictions = [round(x[0]) for x in predictions]

# Saves prediction to a new csv for manual control
mat ={'Real': np.squeeze(np.asarray(label_Column)), 'Predicted': np.squeeze(np.asarray(predictions))}
dataFrame = pa.DataFrame(data=mat)
dataFrame.to_csv("../Data/telco-customer-churn/Predictions.csv")
