import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
import graphviz
from scipy.stats import randint
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.utils.fixes import signature
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
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
scaler.fit(features_test)

features_train = pa.DataFrame(scaler.transform(features_train))
features_train.columns = data.iloc[:, 1:20].columns
features_test = pa.DataFrame(scaler.transform(features_test))
features_test.columns = data.iloc[:, 1:20].columns
print(features_train.head())

def feature_selection(feature_amount, ):
    classifier = LogisticRegression()
    rfe = RFE(classifier, feature_amount)
    fit = rfe.fit(features_train, label_train)
    #print("Selected Features Amount: %d" % (fit.n_features_,))
    #print("Selected Features: %s" % (fit.support_,))
    #print("Feature Ranking: %s" % (fit.ranking_,))
    return fit

def feature_selection_experiment():
    feature_amount_list = [x for x in range(19)]
    selected_features_list = [[] for x in range(19)]
    score_list = [0 for x in range(19)]
    max_predictions = []
    max_accuracy = 0
    for i in range(19):
        feature_amount_list[i] += 1
    for i in range(19):
        temp_features_train = features_train.copy(deep=True)
        temp_features_test = features_test.copy(deep=True)
        temp_fit = feature_selection(feature_amount_list[i])
        j = len(temp_fit.support_) -1
        temp_selected_features = []
        while j >= 0:
            if temp_fit.support_[j] == True:
                temp_selected_features.append(list(data)[j])
            else:
                temp_features_train.drop(labels=str(temp_features_train.columns[j]), axis=1, inplace=True)
                temp_features_test.drop(labels=str(temp_features_test.columns[j]), axis=1, inplace=True)
            j -= 1
        selected_features_list[i] = temp_selected_features
        tree_clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_leaf=86, min_samples_split=89, splitter='best')
        tree_clf.fit(temp_features_train, label_train)
        predictions = tree_clf.predict(temp_features_test)
        accuracy = (accuracy_score(label__test, predictions, normalize=True, sample_weight=None) * 100)
        score_list[i] = accuracy
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_predictions = predictions
    return score_list, selected_features_list, feature_amount_list, max_predictions

scores, selected_features, feature_amounts, max_predictions = feature_selection_experiment()
counter = 0
max_score = 0
max_features = []
max_feature_amount = 0
for feature_list in selected_features:
    print("selected features:" ,feature_list)
    print("score: ", scores[counter])
    print("selected feature count: ", feature_amounts[counter])
    if scores[counter] > max_score:
        max_score = scores[counter]
        max_features = feature_list
        max_feature_amount = feature_amounts[counter]
    counter += 1
print("features that have maximum accuracy: ", max_features)
print("feature count: ", max_feature_amount, " with accuracy:", max_score)
print(confusion_matrix(label__test, max_predictions))
print(classification_report(label__test, max_predictions))
plt.figure(1)
plt.scatter(feature_amounts, scores)
plt.xlabel("Selected Feature Amount")
plt.ylabel("Accuracy")
plt.title("Selected Feature Amount - Accuracy Graph")

fpr,tpr, thresholdsr = roc_curve(label__test, max_predictions)
#roc_auc = auc(labels_test, predictions)
# Plot
plt.figure(2)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1])
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.figure(3)
# Precision Recall Curve
average_precision = average_precision_score(label__test, max_predictions)
precision, recall, _ = precision_recall_curve(label__test, max_predictions)
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

def select_hyperparameters():
    hyperparameter_dict = {"max_depth" : [3,None],
                           "min_samples_leaf" : randint(1,100),
                           "min_samples_split" : randint(2,100),
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
    return tree_cv
#tree_cv = select_hyperparameters()
#print(pa.DataFrame(features_train).head())
# Hyperparameter Tuning
# Setup The Parameters

"""
means = tree_cv.cv_results_['mean_test_score']
stds = tree_cv.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, tree_cv.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()
"""
# Predict the Test set labels with best parameters
#predictions = tree_cv.predict(features_test)
# Evaluations
# Confusion Matrix
#print(confusion_matrix(label__test, predictions))
#print(classification_report(label__test, predictions))
#accuracy = (accuracy_score(label__test, predictions, normalize=True, sample_weight=None) * 100)
#print("Accuracy:" + str(accuracy) + "%")

"""
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

"""