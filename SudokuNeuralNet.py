import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#matplotlib inline
from sklearn.decomposition import PCA

import warnings
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# Read the data
train_data = pd.read_csv('TrainingDigits.csv')
train_data1 = pd.read_csv('TrainingDigits1.csv')
print(1)
train_data = pd.concat([train_data, train_data1],axis =0)
print(2)
train_data2 = pd.read_csv('TrainingDigits2.csv')
train_data = pd.concat([train_data, train_data2],axis =0)
print(3)
train_data3 = pd.read_csv('TrainingDigits3.csv')
print(4)
train_data = pd.concat([train_data, train_data3],axis =0)
test_data = pd.read_csv("SudokuDigits.csv")

train_data = train_data.sample(frac = 1)
# Set up the data
y_train = train_data.iloc[:,0].values
X_train = train_data.drop(train_data.columns[[0]], axis = 1).values/255
# zeroindex = (y_train == 0)
# X_train[y_train == 0,] *= 0.2

X_test = test_data.values


pca_100 = PCA(n_components=20)
pca_100.fit(X_train)
X_train_reduced = pca_100.transform(X_train)
X_test_reduced = pca_100.transform(X_test)

param_grid = parameter_space = {
    'hidden_layer_sizes': [(250,300)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.00025, 0.0002],
    'learning_rate': ['adaptive', 'constant'],
}

mlp = MLPClassifier(max_iter=40, verbose = False)

# mlp = MLPClassifier(
#      hidden_layer_sizes=(250,300),
#      max_iter=80,
#      activation='relu',
#      alpha=0.0004,
#      learning_rate = "constant",
#      solver="adam",
#      verbose=True,
#      random_state=1,
#      #learning_rate_init=0.1
#  )



#mlp.fit(X_train_reduced, y_train)
#We probably won't converge so we'll catch the warnings
with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
      clf = GridSearchCV(mlp, param_grid, n_jobs=1, cv=3, verbose = 3)
      clf.fit(X_train_reduced, y_train)

print('Best parameters found:\n', clf.best_params_)

# # predictions
y_pred_test = clf.predict(X_test_reduced)
guess_board = np.array(y_pred_test).reshape(9,9)
print(guess_board)