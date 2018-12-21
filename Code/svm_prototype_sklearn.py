import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

# Read data, separate features and the labels then split to train and test data
data = pa.read_csv("../Data/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn_Processed.csv")
print(data.head())
features = data.iloc[:, 1:20].copy()
features = np.squeeze(np.asarray(features))

label_Column = data.iloc[:, 20].copy()
# Using Sklearn data split function
features_train, features_test, labels_train, labels_test = train_test_split(features, label_Column, test_size=0.20)

# Feature scaling. Needed since features vary currently a lot
scaler = StandardScaler()
scaler.fit(features_train)
scaler.fit(features_test)

features_train = pa.DataFrame(scaler.transform(features_train))
features_train.columns = data.iloc[:, 1:20].columns

features_test = pa.DataFrame(scaler.transform(features_test))
features_test.columns = data.iloc[:, 1:20].columns

print(features_train.head())
# Feature Selection
def feature_select():
    classifier = LogisticRegression()
    rfe = RFE(classifier, 10)
    fit = rfe.fit(features_train, labels_train)
    print("Num Features: %d") % fit.n_features_
    print("Selected Features: %s") % fit.support_
    print("Feature Ranking: %s") % fit.ranking_
    return fit
'''tmp = feature_select()
i = len(tmp.support_) - 1
while i >= 0:
    if tmp.support_[i] == True:
        print(list(data)[i])
    else:
        features_train.drop(labels=str(features_train.columns[i]), axis=1, inplace=True)
        features_test.drop(labels=str(features_test.columns[i]), axis=1, inplace=True)
    i -= 1
'''
print(pa.DataFrame(features_train).head())
# Param selection
def select_parameters():
    print("Selecting")
    fold_number = 10
    C_list       = [10**x for x in range(-3,3)]
    gamma_list   = [10**x for x in range(-3,3)]
    kernel_list  = ['linear', 'rbf']
    param_dict   = {'C': C_list, 'gamma' :gamma_list, 'kernel': kernel_list}
    grid_search = GridSearchCV(SVC(), param_dict, cv=fold_number, n_jobs=6)

    grid_search.fit(features_train, labels_train)
    return grid_search

#grid_search = select_parameters()
#pa.DataFrame.to_csv(pa.DataFrame(grid_search.cv_results_), "../Data/SVM_grid-search_results.csv")
#print(grid_search.best_params_)
# Best param {'kernel': 'rbf', 'C': 100, 'gamma': 0.001}
# Training
classifier = SVC(kernel='rbf', C=100, gamma=0.001)
classifier.fit(features_train, labels_train)

# Prediction. Predict test data
predictions = classifier.predict(features_test)

# Evaluations
print(confusion_matrix(labels_test, predictions))
print(classification_report(labels_test, predictions))
print("Accuracy:" + str(accuracy_score(labels_test, predictions, normalize=True, sample_weight=None) * 100) + "%")

'''
Manual results:
tenure  InternetService  TotalCharges
First 3 77.8566

tenure  InternetService  OnlineSecurity  Contract  TotalCharges
Firt 5 77.856

tenure  InternetService  OnlineSecurity  TechSupport  StreamingMovies  Contract  TotalCharges
First 7 78.140

tenure  InternetService  OnlineSecurity  TechSupport  StreamingTV  StreamingMovies  Contract  PaperlessBilling  MonthlyCharges  TotalCharges
First 10 79.702

All 80.270
'''
acc_list = [77.857, 77.856, 78.140, 79.702, 80.270]
feature_num_list = [3, 5, 7, 10, 19]

plt.scatter(feature_num_list, acc_list)
plt.show()