
#%% Importing Libs
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from gpdc import GPDC

#%% GENERATE DATA
X1, Y1 = make_blobs(n_samples=1400, center_box=(0, 1), 
                    cluster_std=.08, 
                    n_features=2, # dimension space
                    centers=3, # number of classes 
                    random_state=10)

print('Y1:', set(Y1))

data = pd.DataFrame(data={'x1': X1[:, 0], 'x2': X1[:, 1], 'y': Y1})
sns.scatterplot(data['x1'], data['x2'], hue=data['y'])

#%% TRAIN TEST SPLIT
df_y0_y1 = data.loc[data['y'].isin([0, 1]), :].copy()
X = df_y0_y1.iloc[:,:-1]
y = df_y0_y1.iloc[:,-1]

'''
X_train: X of the classes 0, 1 
y_train: y of the classes 0, 1 -> make all them of the same class 0 
X_test: X of the classes 0, 1, 2
y_test: y of the classes 0, 1, 2
'''

# train
X_train, X_test, y_train, y_test  = train_test_split(X, y, 
                                                      test_size=0.35, random_state=10)
y_train = np.zeros(X_train.shape[0]) # making all train data of the same class 0
print(f'Train Shape {X_train.shape, y_train.shape}')

# test
df_y2 = data.loc[data['y'].isin([2]), :].copy()
X_test = pd.concat([X_test, df_y2.iloc[:,:2]]).reset_index(drop=True) # include class 2
y_test = pd.concat([y_test, df_y2.iloc[:,-1]]).reset_index(drop=True) # include class 2
print(f'Test Shape {X_test.shape, y_test.shape}')

# shuffle test set
from sklearn.utils import shuffle
X_test, y_test = shuffle(X_test, y_test, random_state=10)


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
gpdc = GPDC()
gpdc.fit(X_train, y_train, k=20, alpha=0.1)

xi_l = []
q_negative_l = []
for i, x0 in enumerate(X_train[:2]):
    print(x0)
    
    X_ =
    i=1
    X_train.take(list(range(i))+list(range(i+1, X_train.shape[0])), axis=0)
    xi_hat, R_nk = self._estimate_xi(x0, X_)
    q = self._compute_quantile(xi_hat, R_nk)
    xi_l.append(xi_hat)
    q_negative_l.append(-q)
s = np.quantile(xi_l, 1-alpha/2)
t = np.quantile(q_negative_l, 1-alpha/2)


# R = gpdc._compute_nagated_distances(xa,X1)
# xi_a, u_a = gpdc._estimate_xi(xa, X1)
# print('xi_a,u_a:', xi_a, u_a)
# ya_hat = gpdc.predict(X_test[1])
# print('ya_hat:', ya_hat)
ytest_hat=gpdc.predict(X_test)
print(f'AUC: {roc_auc_score(y_test,ytest_hat)}')

if True:
    plt.figure(figsize=(24, 24))
    plt.scatter(X_train[:,0], X_train[:, 1],marker='.', color='black',s=16)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,marker='x', s=70)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=ytest_hat,marker='o',s=180,alpha=0.5 )
    # plt.scatter(xa[0], xa[1], marker='x', color='green')
    plt.savefig('output/demo.png',dpi = 200,bbox_inches='tight')
    plt.show()

;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   a