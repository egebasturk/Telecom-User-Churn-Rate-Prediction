import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2

# Read data, separate features and the labels then split to train and test data
data = pa.read_csv("dataset.csv")

#cross validation
accuracies = np.zeros((10,4))
trainsetpart = [0.2,0.4,0.6,0.8]
i = 0.2 
cv = 0
index = 0;
rowInd = 0;
while cv < 200:
    cv = cv+20
    print(cv)
    i = 0.2
    rowInd =0
    while i < 1:
        print("train set part " ,i)
        features = data.iloc[:, 2:21].copy()
        features = np.squeeze(np.asarray(features)) #remove 1 dimensions    
        label_Column = data.iloc[:, 21].copy()    
        # Using Sklearn data split function
        features_train, features_test, label_train, label__test = train_test_split(features, label_Column, train_size=i, shuffle = False)   
        #features_selected = SelectKBest(chi2, k=5).fit_transform(features_train,label_train )
        clf = LogisticRegressionCV(cv=cv , random_state=0, multi_class='multinomial').fit(features_train, label_train)
        # Feature scaling. Needed since features vary currently a lot
        #scaler = StandardScaler() #reduce variation
        #scaler.fit(features_train) #compute mean and std
        
        #features_train = scaler.transform(features_train) #standardise by sampling and scaling
        
        #feature selection
        #features_selected_test = np.zeros((2818,5))
        #features_selected_test[:,0] = features_test[:,4];
        #features_selected_test[:,1] = features_test[:,8];
        #features_selected_test[:,2] = features_test[:,14];
        #features_selected_test[:,3] = features_test[:,17];
        #features_selected_test[:,4 ] = features_test[:,18];
        
        #features_test = scaler.transform(features_test)
        
        # Training
        #classifier = LogisticRegression()
        #classifier.fit(features_train, label_train)
        
        # Prediction. Predict test data
        predictions = clf.predict(features_test)    
        # Evaluations
        #print(confusion_matrix(label__test, predictions))
        #print(classification_report(label__test, predictions))
        accuracy = accuracy_score(label__test, predictions)
        print("Accuracy:" + str(accuracy_score(label__test, predictions, normalize=True, sample_weight=None) * 100) + "%")
        accuracies[index][rowInd] =accuracy
        i = i + 0.2
        rowInd =rowInd+1
    index = index+1
    #y_pred_proba = clf.predict_proba(features_test)[:,1]
    #fpr, tpr, _ = metrics.roc_curve(label__test,  y_pred_proba)
    #auc = metrics.roc_auc_score(label__test, y_pred_proba) #area under curve from predictions
    #plt.plot(tpr,label=", True Positives rate")
    #plt.plot(fpr, label=", False Positives rate")
for i in range (10):
    plt.plot(trainsetpart,accuracies[i], label ="cv = %d" %(i*10))
    plt.xlabel("Trainset/Dataset")
    plt.ylabel('Accuracy')
    #plt.legend(loc=4)
    plt.show()