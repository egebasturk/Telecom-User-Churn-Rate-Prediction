import numpy as np
import pandas as pa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

# Read data, separate features and the labels then split to train and test data
data = pa.read_csv("../Data/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn_Processed.csv")

features = data.iloc[:, 2:21].copy()
features = np.squeeze(np.asarray(features))

label_Column = data.iloc[:, 21].copy()
# Using Sklearn data split function
features_train, features_test, label_train, label__test = train_test_split(features, label_Column, test_size=0.25)

"""
row,column = features_train.shape
fs_accuracies = np.zeros((column))
kmaxes = np.zeros((column))
# Feature Selection
for feature_count in range(1, column + 1):
    new_features_train = SelectKBest(chi2, k=feature_count).fit_transform(features_train, label_train)
    new_features_test = SelectKBest(chi2, k=feature_count).fit_transform(features_test, label__test)

    # Feature scaling. Needed since features vary currently a lot
    scaler = StandardScaler()
    scaler.fit(new_features_train)

    new_features_train = scaler.transform(new_features_train)
    new_features_test = scaler.transform(new_features_test)

    krange = 5
    accuracies = np.zeros((krange))
    precision_scores = np.zeros((krange))
    recall_scores = np.zeros((krange))
    f1_scores = np.zeros((krange))

    # Training
    for k in range(1, krange):
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(new_features_train, label_train)

        # Prediction. Predict test data
        predictions = classifier.predict(new_features_test)
        
        # Evaluations
        # print(confusion_matrix(label__test, predictions))
        # print(classification_report(label__test, predictions))
        # print("Accuracy:" + str(accuracy_score(label__test, predictions, normalize=True, sample_weight=None) * 100) + "%")
        
        precision_scores[k] = precision_score(label__test, predictions)
        recall_scores[k] = recall_score(label__test, predictions)
        f1_scores[k] = f1_score(label__test, predictions)
        accuracies[k] = accuracy_score(label__test, predictions, normalize=True, sample_weight=None)

    
    plt.plot(list(range(1, k + 2)), precision_scores)
    plt.xlabel("K value")
    plt.ylabel("Precision Value")
    plt.title("Precision Curve")
    plt.show()

    plt.plot(list(range(1, k + 2)), recall_scores)
    plt.xlabel("K value")
    plt.ylabel("Recall Value")
    plt.title("Recall Curve")
    plt.show()

    plt.plot(list(range(1, k + 2)), f1_scores)
    plt.xlabel("K value")
    plt.ylabel("F1-Score")
    plt.title("F1-Score Curve")
    plt.show()

    plt.plot(list(range(1, k + 2)), accuracies)
    plt.xlabel("K value")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Curve")
    plt.show()
    
    k_final = np.argmax(accuracies)
    kmaxes[feature_count - 1] = k_final
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(new_features_train, label_train)
    predictions = classifier.predict(new_features_test)
    fs_accuracies[feature_count - 1] = accuracy_score(label__test, predictions, normalize=True, sample_weight=None)


plt.plot(list(range(1, column + 1)), fs_accuracies)
plt.xlabel("Feature Count")
plt.ylabel("Test Accuracy")
plt.title("Feature Selection Test Accuracy Plot")
plt.show()
"""

# Final Output
feature_count = 6
k = 4

print("Feature Count: " + str(feature_count))
print("K: " + str(k))

selector = SelectKBest(chi2, k=feature_count)
features_train = selector.fit_transform(features_train, label_train)
features_test = selector.fit_transform(features_test, label__test)

cols = selector.get_support(indices=True)
print("Selected Features: ")
print(cols)

classifier = KNeighborsClassifier(n_neighbors=k)
classifier.fit(features_train, label_train)
predictions = classifier.predict(features_test)
print("Confusion Matrix: ")
print(confusion_matrix(label__test, predictions))
print(classification_report(label__test, predictions))
print("Accuracy: " + str(accuracy_score(label__test, predictions, normalize=True, sample_weight=None)))

"""
fpr, tpr, _ = roc_curve(label__test, predictions)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

precision, recall, _ = precision_recall_curve(label__test, predictions)
average_precision = average_precision_score(label__test, predictions)
step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()


import numpy as np
x = np.asarray(["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"])
indices = np.asarray([4, 8, 11, 14, 17, 18])
features_selected = [x[i] for i in indices]
print(features_selected)
"""