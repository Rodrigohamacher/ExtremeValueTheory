import numpy as np
from scipy.stats import genextreme


class GPDC:
    '''
    The GPDC classifier.
    '''

    def _compute_nagated_distances(self, x0, X):
        '''
        x0:1 x p vector
        X :n x p matrix
        '''
        x0 = x0.reshape(1, -1)
        assert X.shape[1] == x0.shape[1], 'Wrong dimension of the new point!'
        D = np.sqrt(np.sum((X-x0)**2, axis=1))
        return -D

    def _estimate_xi(self, x0, X):
        k = self.k
        R = self._compute_nagated_distances(x0, X)
        R_selected = sorted(R)[-k:]
        # print('R_selected', R_selected)
        u = sorted(R)[-(k+1)]
        assert u != 0
        xi_hat = np.sum(np.log(R_selected/u))/k
        return xi_hat, u

    def _compute_quantile(self, xi_hat, R_nk):
        '''
        Get (1-1/n)-quantile of R by R_(n-k)+H^(-1)(1-1/k)
        '''
        q = R_nk*self.k**xi_hat
        return q

    def _compute_thresholds(self, alpha):
        '''
        Set thresholds (s,t) to the (1-alpha/2) quantiles of the (xi^(1),...xi^(n)) and (-q^(1),...,-q^(n))
        '''
        xi_l = []
        q_negative_l = []
        for i, x0 in enumerate(self.X):
            X_ = self.X.take(list(range(i))+list(range(i+1, self.n)), axis=0)
            xi_hat, R_nk = self._estimate_xi(x0, X_)
            q = self._compute_quantile(xi_hat, R_nk)
            xi_l.append(xi_hat)
            q_negative_l.append(-q)
        s = np.quantile(xi_l, 1-alpha/2)
        t = np.quantile(q_negative_l, 1-alpha/2)
        # print('s,t:', s, t)
        return s, t

    def fit(self, X, y, k, alpha):
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