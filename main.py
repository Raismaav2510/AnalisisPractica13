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

model = LogisticRegression(random_state=0).fit(X_train, Y_train)
print('Training score: ', f1_score(Y_train, model.predict(X_train)))
print('Testing score: ', f1_score(Y_test, model.predict(X_test)))


def correlation_matrix(Y, X, is_plot=False):
    # Calculate and plot the correlation symmetrical matrix
    # Return: yX - concatenated data
    # yX_corr - correlation matrix, pearson correlation of values from -1 to +1
    # yX_abs_corr - correlation matrix, absolute values
    yX = pd.concat([Y, X], axis=1)
    yX = yX.rename(columns={0: 'TARGET'})  # Rename first column
    print('Function correlation_matrix: X.shape, Y.shape, yX.shape: ', X.shape, Y.shape, yX.shape)
    print()
    # Get feature correlations and transform to dataframe
    yX_corr = yX.corr(method='pearson')
    # Convert to absolute values
    yX_abs_corr = np.abs(yX_corr)
    if is_plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(yX_abs_corr, cmap='RdYlGn', interpolation='none', aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(yX_abs_corr)), yX_abs_corr.columns, rotation='vertical')
        plt.yticks(range(len(yX_abs_corr)), yX_abs_corr.columns)
        plt.title('Pearson Correlation Heat Hap (absolute values)', fontsize=15, fontweight='bold')
        plt.show()
        return yX, yX_corr, yX_abs_corr

    # Build the correlation matrix for the train data
    yX, yX_corr, yX_abs_corr = correlation_matrix(Y, X, is_plot=True)


# Applying PCA
pca = PCA()
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)
# Plotting the amount of information stored in each component
plt.ylabel('Variance')
plt.xlabel('Component Number')
plt.bar(np.arange(30) + 1, pca.explained_variance_ratio_)
plt.show()

pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)
X_train_reduced_malignant = np.array([X for X, Y in zip(X_train_reduced, Y_train) if Y == 0])
X_train_reduced_benign = np.array([X for X, Y in zip(X_train_reduced, Y_train) if Y == 1])
plt.scatter(*X_train_reduced_malignant.T, color='red')
plt.scatter(*X_train_reduced_benign.T, color='blue')
plt.title('Training Set After PCA')
plt.legend(['malignant', 'benign'])
plt.xlabel('Coordinate of first principle component')
plt.ylabel('Coordinate of second principle component')
plt.show()

pca = PCA(n_components=5)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)
model = LogisticRegression(random_state=0).fit(X_train_reduced, Y_train)
print('Training score: ', f1_score(Y_train, model.predict(X_train_reduced)))
print('Testing score: ', f1_score(Y_test, model.predict(X_test_reduced)))

X = np.arange(30) + 1
Y = []
for i in X:
    pca = PCA(n_components=i)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    model = LogisticRegression(random_state=0).fit(X_train_reduced, Y_train)
    Y.append(f1_score(Y_train, model.predict(X_train_reduced)))
plt.plot(X, Y)
plt.xlabel('Number of Components')
plt.ylabel('Training Score')
plt.show()

Y = []
for i in X:
    pca = PCA(n_components=i)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    model = LogisticRegression(random_state=0).fit(X_train_reduced, Y_train)
    Y.append(f1_score(Y_test, model.predict(X_test_reduced)))
plt.plot(X, Y)
plt.xlabel('Number of Components')
plt.ylabel('Test Score')
plt.show()

