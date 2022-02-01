
#%% Importing Libs
from typing import Tuple, Union

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


def df_preprocess(df: Union[pd.DataFrame, np.array],
                  normal_classes: Union[pd.Series, np.array],
                  abnormal_classes: Union[pd.Series, np.array],
                  test_size: float = 0.35,
                  random_state: int = 10):
    '''
    
    It is implied that Y is the last column of the DataFrame df
    
    Parameters
    ----------
    df: pd.DataFrame, np.array
    normal_classes: pd.Series, np.array
    abnormal_classes: pd.Series, np.array
    test_size: float
 
    Return
    ----------
    X_train, X_test, y_train, y_test

    '''
    
    coly = df.columns[-1]
    
    # Training Samples
    df_normal = df.loc[df[coly].isin(normal_classes), :].copy() 
    X_normal = df_normal.iloc[:,:-1]
    y_normal = df_normal.iloc[:,-1]
    
    # Split dataset into X,y using only the training labels
    # After data, append the new classes to y_test and X_test
    X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(X_normal,
                                                                                    y_normal, 
                                                    test_size=test_size,
                                                    random_state=random_state)

    # TRAIN -> only normal class
    # X_train: X of the classes 0, 1 
    # y_train: y of the classes 0, 1 -> make all them of the same class => 0 
    # making all train data of the same class 0
    y_normal_train[:] = 0 

    # TEST -> normal classs + abnormal
    # X_valid: X of the classes 0, 1, 2
    # y_valid: y of the classes 0, 1, 2 -> make class 0 and 1 being => 0 and class 2 being => 1
    y_normal_test[:] = 0 # making all train data of the same class 0

    df_abnormal = df.loc[df[coly].isin(abnormal_classes), :].copy() # get only class 2
    df_abnormal[coly] = 1 # class 2 being => 1

    X_abnormal_test = df_abnormal.iloc[:,:-1]
    y_abnormal_test = df_abnormal.iloc[:,-1] 
    
    X_test = pd.concat([X_normal_test, X_abnormal_test]
                    ).reset_index(drop=True) # include class 1 at the end of X_test
    
    y_test = pd.concat([y_normal_test, y_abnormal_test]
                    ).reset_index(drop=True) # include class 1 at the end of y_test

    from sklearn.utils import shuffle
    X_test, y_test = shuffle(X_test, y_test, random_state=random_state) # shuffle data

    X_normal_train = X_normal_train.values
    X_test = X_test.values
    
    return X_normal_train, X_test, y_normal_train, y_test

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
Split dataset into X,y using only the training labels
After data, append the new classes to y_test and X_test

TRAIN -> only normal class
X_train: X of the classes 0, 1 
y_train: y of the classes 0, 1 -> make all them of the same class => 0 

TEST -> normal classs + abnormal
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

X_train2, X_test2, y_train2, y_test2 = df_preprocess(data, normal_classes = [0,1], abnormal_classes = [2]) 
suggested_alpha = 1/X_train2.shape[1]
gpdc = GPDC(k=20, alpha=1/10)
gpdc.fit(X_train2, y_train2)
ytest_hat2 = gpdc.predict(X_test2)    
evaluate_compilation(y_test2, ytest_hat2)

#%% GENERATE DATA - 8.2 OLETTER protocol
filename = '../dataset/OLETTER/letter-recognition.data'
oletter = pd.read_csv(filename,sep=',', header=None)
oletter.columns = ['lettr','x-box', 'y-box','width', 'high', 'onpix', 'x-bar', 'y-bar',
           'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 
           'yegvx']

reorder_cols = oletter.columns[1:].tolist() + [oletter.columns[0]] 
oletter = oletter.loc[:,reorder_cols]
oletter.shape

'''
	1. lettr	capital letter	(26 values from A to Z)
	 2.	x-box	horizontal position of box	(integer)
	 3.	y-box	vertical position of box	(integer)
	 4.	width	width of box			(integer)
	 5.	high 	height of box			(integer)
	 6.	onpix	total # on pixels		(integer)
	 7.	x-bar	mean x of on pixels in box	(integer)
	 8.	y-bar	mean y of on pixels in box	(integer)
	 9.	x2bar	mean x variance			(integer)
	10.	y2bar	mean y variance			(integer)
	11.	xybar	mean x y correlation		(integer)
	12.	x2ybr	mean of x * x * y		(integer)
	13.	xy2br	mean of x * y * y		(integer)
	14.	x-ege	mean edge count left to right	(integer)
	15.	xegvy	correlation of x-ege with y	(integer)
	16.	y-ege	mean edge count bottom to top	(integer)
	17.	yegvx	correlation of y-ege with x	(integer)
'''
oletter['lettr'].value_counts()

classes = list(set(oletter['lettr']))
len(classes)


random_state = 12
np.random.seed(random_state)
training_classes = np.random.choice(classes, 15, replace=False)
len(training_classes)
abnormal_classes = set(classes).difference(set(training_classes))
len(abnormal_classes)
X_train, X_test, y_train, y_test = df_preprocess(oletter, 
                                                 normal_classes = training_classes,
                                                 abnormal_classes = abnormal_classes,
                                                 test_size=0.1,
                                                 random_state=random_state)
y_train.value_counts()
X_train.shape
suggested_alpha = 1/X_train.shape[1]
gpdc = GPDC(k=20, alpha=0.01)
gpdc.fit(X_train, y_train)
ytest_hat = gpdc.predict(X_test)
evaluate_compilation(y_test, ytest_hat)

