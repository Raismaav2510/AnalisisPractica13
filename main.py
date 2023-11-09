import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dat = datasets.load_breast_cancer()
print(dat.DESCR)

df_all = pd.DataFrame(dat['data'], columns=list(dat['feature_names']))
print(df_all.head())

TEST_SIZE_RATIO = 0.3
x = df_all
y = pd.Series(list(dat['target']))
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=TEST_SIZE_RATIO, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print('X_train.shape, Y_train.shape', X_train.shape, Y_train.shape)
print('X_test.shape, Y_test.shape', X_test.shape, Y_test.shape)
