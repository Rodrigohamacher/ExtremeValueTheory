import numpy as np
from scipy.stats import genextreme


class GEVC:
    '''
    The GEV classifier
    '''
    def _max_negated_distances(self, x0, X):
        '''
        x0:1 x p vector
        X :n x p matrix
        ---
        Consider Euclidean distcance for simplicity.
        '''
        x0 = x0.reshape(1, -1)
        print(x0.shape, X.shape, 'A test!')
        assert X.shape[1] == x0.shape[1], 'Wrong dimension of the new point!'
        D = np.sqrt(np.sum((X-x0)**2, axis=1))
        max_negated_D = np.max(-D)
        return max_negated_D

    def _estimate_xi(self, X):
        '''
        Fit a reversed Weibull distribution for the maximum of the negated distances between a known point and all the remaining points in the training set. 
        Estimate the parameters ($\xi$, $\mu$, $\sigma$) for Weibull distribution.
        '''
        nd_list = []
        for i, row in enumerate(X):
            X_ = np.delete(X, i, axis=0)
            # nd=np.max(self._compute_negated_distances(row,X_))
            nd = self._max_negated_distances(row, X_)
            nd_list.append(nd)

        xi, mu, sigma = genextreme.fit(nd_list)
        return xi, mu, sigma

    def fit(self, X, y, alpha):
        self.alpha = alpha
        self.X = X
        self.n, self.p = X.shape
        self.xi, self.mu, self.sigma = self._estimate_xi(X)

    def _predict(self, x0,prob):
        '''
        Perform the hypothesis test: 
            H_0: x0 is known.
            H_a: x0 is unknown.
        Reject H_0 if W_hat(mnd0)<alhpa
        ---
        input: x0 <- new test point;
               prob <- logical indicating whether the probability that x0 is unknown should be returned (1 - p-value).
        return: 1 -> unknown point detected.
        '''
        mnd0 = self._max_negated_distances(x0, X)
        p = genextreme.cdf(x=mnd0, c=self.xi, loc=self.mu, scale=self.sigma)
        if prob:
            return 1-p
        else:
            return 1 if p < self.alpha else 0

    def predict(self, X_test,prob=True):
        y_hat = np.fromiter((self._predict(x0,prob) for x0 in X_test), X_test.dtype)
        return y_hat
