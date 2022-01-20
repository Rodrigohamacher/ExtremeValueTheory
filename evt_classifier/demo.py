
#%% Importing Libs
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef

from gpdc import GPDC

#%% FUNCTIONS
def evaluate_compilation(y_test, y_pred):
    roc = roc_auc_score(y_test,y_pred)
    acc = balanced_accuracy_score(y_test, y_pred)
    # The Matthews correlation coefficient (+1 represents a perfect prediction, 
    # 0 an average random prediction and -1 and inverse prediction).
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f'AUC: {roc}')
    print(f'acc {acc.round(3)}, mcc {mcc.round(3)}')


#%% GENERATE DATA - 8.1 Simulated data
X1, Y1 = make_blobs(n_samples=1400, center_box=(0, 1), 
                    cluster_std=.08, 
                    n_features=2, # dimension space
                    centers=3, # number of classes 
                    random_state=10)

print('Y1:', set(Y1))

data = pd.DataFrame(data={'x1': X1[:, 0], 'x2': X1[:, 1], 'y': Y1})
sns.scatterplot(data['x1'], data['x2'], hue=data['y'])

#%% TRAIN TEST SPLIT
# only classes 0, 1 for training data
df_y0_y1 = data.loc[data['y'].isin([0, 1]), :].copy() 
X = df_y0_y1.iloc[:,:-1]
y = df_y0_y1.iloc[:,-1]

'''
X_train: X of the classes 0, 1 
y_train: y of the classes 0, 1 -> make all them of the same class => 0 
X_test: X of the classes 0, 1, 2
y_test: y of the classes 0, 1, 2 -> make class 0 and 1 being => 0 and class 2 being => 1
'''

#%% train
X_train, X_test, y_train, y_test  = train_test_split(X, y, 
                                                    test_size=0.35,
                                                    random_state=10)
y_train[:] = 0 # making all train data of the same class 0
print(f'Train Shape {X_train.shape, y_train.shape}')

#%% test
y_test[:] = 0 # making all train data of the same class 0
print(f'Test Shape {X_test.shape, X_test.shape}')

df_y2 = data.loc[data['y'].isin([2]), :].copy() # get only class 2
df_y2['y'] = 1 # class 2 being => 1

X_test = pd.concat([X_test, df_y2.iloc[:,:2]]
                   ).reset_index(drop=True) # include class 1 at the end of X_test
y_test = pd.concat([y_test, df_y2.iloc[:,-1]]
                   ).reset_index(drop=True) # include class 1 at the end of y_test
print(f'Test Shape {X_test.shape, y_test.shape}')

#%% shuffle test set
from sklearn.utils import shuffle
X_test, y_test = shuffle(X_test, y_test, random_state=10) # shuffle data

X_train = X_train.values
X_test = X_test.values

# if False:
plt.figure(figsize=(8, 8))
plt.scatter(X_train[:,0], X_train[:, 1],marker='.', c=y_train, s=25, edgecolor='k')
plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', c=y_test)
plt.legend(['training_set', 'test_set'])
# plt.scatter(xa[0], xa[1], marker='x', color='green')
plt.show()

#%% MODEL
suggested_alpha = 1/X.shape[1]
gpdc = GPDC(k=20, alpha=1/10)
gpdc.fit(X_train, y_train)
ytest_hat = gpdc.predict(X_test)    
evaluate_compilation(y_test, ytest_hat)

plt.figure(figsize=(24, 24))
plt.scatter(X_train[:,0], X_train[:, 1],marker='.', color='black',s=16)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,marker='x', s=70)
plt.scatter(X_test[:, 0], X_test[:, 1], c=ytest_hat,marker='o',s=180,alpha=0.5 )
# plt.scatter(xa[0], xa[1], marker='x', color='green')
plt.savefig('output/demo.png',dpi = 200,bbox_inches='tight')
plt.show()


#%% GENERATE DATA - 8.2 OLETTER protocol
