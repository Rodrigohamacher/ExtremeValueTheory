from typing import Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import genextreme


class GPDC:
    '''
    The GPDC classifier.
    '''

    def _compute_nagated_distances(self, x0: list, X: list) -> list:
        '''
        Calculates all the [R(i)] negated distances between x0 
        and all the other observations of the dataset
        
        Parameters
        ----------
        x0: float
            x0:1 x p vector
            x0 are the values of a specific row on the dataset
        X: float
            X :n x p matrix
            X is the remaning observations of the dataset, except the x0 observation

        Returns
        ----------
        It returns [R(i)] a numpy array list of the negated distances (squared difference)
        between x0 and all the other observations of the dataset 
        '''
        x0 = x0.reshape(1, -1)
        # verifica se possuem a mesma qntd de features
        assert X.shape[1] == x0.shape[1], 'Wrong dimension of the new point!'
        D = np.sqrt(np.sum((X-x0)**2, axis=1))
        return -D

    def _estimate_xi(self, x0: list, X: list) -> Tuple[float, list]:
        '''
        Calculates and selects the [R(n+1-i) list] K highest negated distances 
        and the [R(n-k)] K+1 highest negated distance between x0 and X.
        Besides that, it calculates the Csi (ξ) of the x0 observation  
        
        Parameters
        ----------
        x0: float
            x0:1 x p vector
            x0 are the values of a specific row on the dataset
        X: float
            X :n x p matrix
            X is the remaning observations of the dataset, except the x0 observation
        
        Returns
        ----------
        It returns the [R(n-k)] K+1 smallest negated distance 
        between x0 and X and Csi (ξ) for x0.

        '''
                
        k = self.k # k highest distances [-0.01 > -3.7 > -11]
        # all the negated distances between x0 and all the other observations
        R = self._compute_nagated_distances(x0, X) 
        # Select the K highest negated distances
        R_selected = sorted(R)[-k:] # --> R(n+1-i) list
        # Select the K+1 highest negated distances
        u = sorted(R)[-(k+1)] # --> R(n-k)
        assert u != 0
        # calculates Csi (ξ) of the x0
        xi_hat = np.sum(np.log(R_selected/u))/k
        return xi_hat, u

    def _compute_quantile(self, xi_hat, R_nk, alpha: float) -> float:
        '''
        Get (1-1/n)-quantile of R by R_(n-k)+H^(-1)(1-1/k) = R(n-k)(n*alpha/k)^-ξ
        In this case, alpha
        
        Parameters
        ----------
        xi_hat: float
            (1-alpha)-quantile of the negated_distance
            Suggestion, choose alpha = 1/n, where n is the number of rows for X_train
            
        Returns
        ----------
        
        '''
        # q = R_nk*self.k**xi_hat
        q = R_nk * (self.n * alpha/self.k) ** -xi_hat
        return q

    def _compute_thresholds(self, alpha: float):
        '''
        Set thresholds (s,t) to the (1-alpha/2) quantiles 
        of the (xi^(1),...xi^(n)) and (-q^(1),...,-q^(n))
        
        Parameters
        ----------
        alpha: float
            (1-alpha)-quantile of the negated_distance
            Suggestion, choose alpha = 1/n, where n is the number of rows for X_train
            
        Returns
        ----------
        
        '''
        xi_l = []
        q_negative_l = []
        for i, x0 in enumerate(self.X):
            # x0 is the i th row
            # return all the observations except the i th observation
            X_ = self.X.take(list(range(i))+list(range(i+1, self.n)), axis=0)
            # for each x0 i th row observation, it will have an associated distribution
            
            # that considers the distance of x0 and all the remaining observations
            # It returns the [R(n-k)] K+1 highest negated distance 
            # between x0 and X and Csi (ξ) for x0.
            xi_hat, R_nk = self._estimate_xi(x0, X_)
            
            # *********** PAREI AQUI 09/01/2022 **************
            q = self._compute_quantile(xi_hat, R_nk, alpha)

            xi_l.append(xi_hat)
            q_negative_l.append(-q)

        s = np.quantile(xi_l, 1-alpha/2)
        t = np.quantile(q_negative_l, 1-alpha/2)
        # print('s,t:', s, t)
        return s, t

    def fit(self,
            X: Union[pd.DataFrame, list], 
            y: Union[pd.DataFrame, list], 
            k: int,  
            alpha: float):
        
        """
        Train model
        
        Parameters
        ----------
        X: pd.DataFrame or np.array
            X_train 
        y: pd.DataFrame or np.array
            y_train
        k: int
            k higher distances
        alpha: float
            (1-alpha)-quantile of the negated_distance
            Suggestion, choose alpha = 1/n, where n is the number of rows for X_train
        """
        
        assert 0 < alpha < 1, 'Wrong value of alpha!'
        self.k = k
        self.n, self.p = X.shape
        self.X = X
        self.y = y
        # compute thresholds
        s, t = self._compute_thresholds(alpha)
        assert s > -1, 'Wrong value of the estimated s!'
        assert t > 0, 'Wrong value of the estimated t!'
        self.s = s
        self.t = t

    def _predict(self, x0):
        '''
        return: 1 -> unknown point detected.
        '''
        xi_hat, R_nk = self._estimate_xi(x0, self.X)
        # First test
        if xi_hat*self.p > self.s:
            return 1
        else:
            # Second test
            q = self._compute_quantile(xi_hat, R_nk)
            if -q > self.t:
                return 1
            else:
                return 0

    def predict(self, X_test):
        y_hat = np.fromiter((self._predict(x0) for x0 in X_test), X_test.dtype)
        return y_hat