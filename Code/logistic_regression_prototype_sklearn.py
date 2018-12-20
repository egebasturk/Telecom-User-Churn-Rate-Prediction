import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
import graphviz
from scipy.stats import randint
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, average_precision_score
from sklearn.utils.fixes import signature

# Read data, separate features and the labels then split to train and test data
data = pa.read_csv("../Data/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn_Processed.csv")

features = data.iloc[:, 1:20].copy()
features = np.squeeze(np.asarray(features))

label_Column = data.iloc[:, 20].copy()
# Using Sklearn data split function
features_train, features_test, label_train, label__test = train_test_split(features, label_Column, test_size=0.40)

# Feature scaling. Needed since features vary currently a lot
scaler = StandardScaler()
scaler.fit(features_train)

features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

# Hyperparameter Tuning
# Setup The Parameters
hyperparameter_dict = {"max_depth" : [3,None],
                       "min_samples_leaf" : randint(1,9),
                       "criterion" : ["gini", "entropy"],
                       "splitter" : ["best", "random"]}
# Decision Tree Classifier
tree_clf = DecisionTreeClassifier()

# Randomized Search Cross Validation
tree_cv = RandomizedSearchCV(tree_clf, hyperparameter_dict, cv = 5)

# Fit data to CV
tree_cv.fit(features_train, label_train)

# Check Tuned Hyperparameters and Scores
print("Tuned Decision Tree Hyperparameters: {}".format(tree_cv.best_params_))
print("Best Score: {}".format(tree_cv.best_score_))

# Predict the Test set labels with best parameters
predictions = tree_cv.predict(features_test)

# Evaluations
# Confusion Matrix
print(confusion_matrix(label__test, predictions))
print(classification_report(label__test, predictions))
accuracy = (accuracy_score(label__test, predictions, normalize=True, sample_weight=None) * 100)
print("Accuracy:" + str(accuracy) + "%")

# Tree Visualization
iris = load_iris()
clf = DecisionTreeClassifier()
clf.fit(iris.data, iris.target)
dot_data = tree.export_graphviz(clf, out_file=None,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(dot_data)

# Precision Recall Curve
average_precision = average_precision_score(label__test, predictions)
precision, recall, _ = precision_recall_curve(label__test, predictions)
step_kwargs = ({'step':'post'}
                if 'step' in signature(plt.fill_between).parameters
                else {})
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()