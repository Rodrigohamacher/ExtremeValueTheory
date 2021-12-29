
#%% Importing Libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from gpdc import GPDC


#%% RUN
X1, Y1 = make_blobs(n_samples=1400,\
    center_box=(0, 1), cluster_std=.08, n_features=2, centers=3, random_state=10)
# print('Y1:', set(Y1))
data = pd.DataFrame(data={'x1': X1[:, 0], 'x2': X1[:, 1], 'y': Y1})
test_1 = data[data['y'] == 2]
train, test_2 = train_test_split(data.loc[data['y'].isin([0, 1]), :], test_size=0.35)
test = pd.concat([test_1, test_2]).reset_index(drop=True)

X_train,y_train=train.values[:,:2],train.values[:,2]
y_train=np.zeros(train.shape[0])

X_test=test.values[:,:2]
y_test=test.values[:,2].reshape(1,-1).squeeze()
y_test=np.array(list(map(lambda x_:0 if x_ in (0,1) else 1,y_test)))
# xa = X1[0]+np.array([1, 1])
# print(train.shape,test.shape)
if False:
    plt.figure(figsize=(8, 8))
    plt.scatter(X_train[:,0], X_train[:, 1],marker='.', c=y_train, s=25, edgecolor='k')
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', c=y_test)
    # plt.scatter(xa[0], xa[1], marker='x', color='green')
    plt.show()

gpdc = GPDC()
gpdc.fit(X_train, y_train, k=20, alpha=0.1)

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

