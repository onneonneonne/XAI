from ucimlrepo import fetch_ucirepo # My mushroom dataset
import sklearn
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import shap
import copy
np.random.seed(1)

# Get dataset 
mushroom = fetch_ucirepo(id=73) 

# Get features
X = pd.DataFrame(mushroom.data.features)

# Encode features (from categorical to integers)
encoders = {} 
X = X.replace('?', np.nan)
X = X.fillna(X.mode().iloc[0]) # Missing value handling
X_encoded = X.copy()
for col in X.columns:
    enc = LabelEncoder()
    X_encoded[col] = enc.fit_transform(X[col])
    encoders[col] = enc

# Get targets
y = pd.DataFrame(mushroom.data.targets)
le = LabelEncoder()
y_encoded = y.apply(le.fit_transform)

# Split train-test
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=6)

# Train model
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=1000, random_state=42)
rf.fit(train, labels_train.values.ravel())

# Accuracy
y_pred = rf.predict(test)
print(f"Accuracy RF: {100*accuracy_score(labels_test, y_pred)}%")

def SHAP(waterfall_list):
    explainer = shap.TreeExplainer(rf)

    # Set up model
    nb_points_explain = round(0.2*train.shape[0])
    shap_values = explainer(train.iloc[0:nb_points_explain, :])

    # Relevant dimention
    sv = shap_values[:, :, 1]
    sv.data = train.iloc[0:nb_points_explain, :].copy()

    # De-encode
    sv_encoded = copy.deepcopy(sv)
    for col in X.columns:
        sv.data[col] = encoders[col].inverse_transform(sv.data[col].astype(int))

    # Waterfall plot for single explanations:
    for i in waterfall_list:
        shap.plots.waterfall(sv[i])

    # Beeswarm (aka summary plot)
    shap.plots.beeswarm(sv_encoded)

    # Barplot
    shap.plots.bar(sv)

    # Heatmap
    shap.plots.heatmap(sv)


def PI(n):
    # Compute permutation importance
    r = permutation_importance(rf, test, labels_test.values.ravel(), n=10, random_state=0)

    # Sort r
    sorted_r = r.importances_mean.argsort()

    # Print permutation importance
    print("Permutation Importance:")
    for i in sorted_r[::-1]:
        print(f"{test.columns[i]}: {r.importances_mean[i]:.4f}")

    # Bar plot
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(test.columns)[sorted_r], r.importances_mean[sorted_r])
    plt.xlabel("Mean accuracy decrease")
    plt.title(f"Permutation Importance (n_repeats={n})")
    plt.show()

SHAP([0,2])
PI(10)