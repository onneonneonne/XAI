import pandas as pd
from ucimlrepo import fetch_ucirepo # My mushroom dataset

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import shap

  
# fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 
  
# metadata 
print(mushroom.metadata) 
  
# variable information 
print(mushroom.variables) 

# df = pd.read_csv('data/heart.csv')
# X = df.drop(['target'],axis=1)
# y = df['target']

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(X, y, test_size=1, random_state=7)

print(train.head())

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=1000, random_state=42)
rf.fit(train, labels_train)

y_pred = rf.predict(test)
print("Accuracy: ", accuracy_score(labels_test, y_pred))

# SHAP explanations
explainer = shap.KernelExplainer(rf.predict, shap.kmeans(train, 10))
nb_points_explain = round(0.2*train.shape[0])
shap_values = explainer(train.iloc[0:nb_points_explain, :])

ref = explainer.expected_value
print("Average predicted output: ",ref)

# Waterfall plot for a single explanation
i = 3
shap.plots.waterfall(shap_values[i])

i = 2
shap.plots.waterfall(shap_values[i])