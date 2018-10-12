import os
# Backend change
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import backend as K
import pandas as pa

data = pa.read_csv("../Data/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn_Processed.csv")
#print(data)
features = data.iloc[:, 1:20].copy()
#print(features)
target_Column = data.iloc[:, 21].copy()
print(target_Column)