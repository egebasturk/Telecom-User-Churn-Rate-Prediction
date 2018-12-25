'''
@author Alp Ege Basturk
Final Neural Network model
'''
import os
# Backend change
os.environ['KERAS_BACKEND'] = 'theano' # Comment out if using Tensodflow
#os.environ['THEANO_FLAGS'] = 'device=opencl1:0, floatX=float32'

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression

# This reads data from the csv and save it to a data frame object
data = pa.read_csv("../Data/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn_Processed.csv")
#print(data)
# following separate features and the labels
features = data.iloc[:, 1:20].copy()
features = np.squeeze(np.asarray(features))
#print(features)
label_Column = data.iloc[:, 20].copy()

# Using Sklearn data split function
features_train, features_test, labels_train, labels_test = train_test_split(features, label_Column, test_size=0.40)

# Feature scaling. Needed since features vary currently a lot
scaler = StandardScaler()
scaler.fit(features_train)

features_train = pa.DataFrame(scaler.transform(features_train))
features_train.columns = data.iloc[:, 1:20].columns

features_test = pa.DataFrame(scaler.transform(features_test))
features_test.columns = data.iloc[:, 1:20].columns


print(features_train.head())
# Feature Selection
# Change number for best i features
feature_count = 10
def feature_select():
    classifier = LogisticRegression()
    rfe = RFE(classifier, feature_count)
    fit = rfe.fit(features_train, labels_train)
    print("Num Features: %d") % fit.n_features_
    print("Selected Features: %s") % fit.support_
    print("Feature Ranking: %s") % fit.ranking_
    return fit

tmp = feature_select()
i = len(tmp.support_) - 1
while i >= 0:
    if tmp.support_[i] == True:
        print(list(data)[i])
    else:
        features_train.drop(labels=str(features_train.columns[i]), axis=1, inplace=True)
        features_test.drop(labels=str(features_test.columns[i]), axis=1, inplace=True)
    i -= 1

print(pa.DataFrame(features_train).head())


# Model creation
# tells keras to create the model
model = Sequential()

model.add(Dense(20, input_dim=feature_count, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# Model compiliation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model fitting
history = model.fit(features_train, labels_train, epochs=100, batch_size=5, verbose=1)

# Evaluate the model
scores = model.evaluate(features_test, labels_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Predict
predictions = model.predict(features_test)
# Round predictions
predictions = [round(x[0]) for x in predictions]

# Saves prediction to a new csv for manual control
mat ={'Real': np.squeeze(np.asarray(labels_test)), 'Predicted': np.squeeze(np.asarray(predictions))}
dataFrame = pa.DataFrame(data=mat)
dataFrame.to_csv("../Data/telco-customer-churn/Predictions.csv")

print(confusion_matrix(labels_test, predictions))
print(classification_report(labels_test, predictions))
acc = accuracy_score(labels_test, predictions, normalize=True, sample_weight=None)
print("Accuracy:" + str(acc * 100) + "%")


plt.subplot(1,2,1)
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.axhline(y=acc, color='r')
plt.ylim(0.4,1)

#Complex one
model2 = Sequential()
model2.add(Dense(200, input_dim=feature_count, activation='relu'))
model2.add(Dense(150, activation='relu'))
model2.add(Dense(100, activation='relu'))
model2.add(Dense(50, activation='relu'))
model2.add(Dense(25, activation='relu'))
model2.add(Dense(1,activation='sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history2 = model2.fit(features_train, labels_train, epochs=100, batch_size=5, verbose=1)
scores = model2.evaluate(features_test, labels_test)
predictions2 = model2.predict(features_test)
predictions2 = [round(x[0]) for x in predictions2]
acc = accuracy_score(labels_test, predictions2, normalize=True, sample_weight=None)

plt.subplot(1,2,2)
plt.plot(history2.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.axhline(y=acc, color='r')
plt.ylim(0.4,1)
plt.show()