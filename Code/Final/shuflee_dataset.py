# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 19:08:21 2018

@author: albam
"""
from sklearn.utils import shuffle
import pandas as pa

# Read data, separate features and the labels then split to train and test data
data = pa.read_csv("../Data/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn_Processed.csv")

dataset = data.iloc[:,1:22].copy()
dataset = shuffle(dataset)

dataset.to_csv('dataset.csv')