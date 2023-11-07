import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dat = datasets.

df_all = pd.DataFrame(dat['data'], columns=list(dat['feature_names']))

TEST_SIZE_RATIO = 0.3

x = df_all
y = pd.Series(list(dat['target']))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE_RATIO, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
