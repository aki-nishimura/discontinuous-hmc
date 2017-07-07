import pandas as pd
import numpy as np

### Load and clean the SECOM data.
secom = pd.read_table('../data_and_posterior/secom_features.txt', sep='\s+', header=None)
y = pd.read_table('../data_and_posterior/secom_outcome.txt', sep='\s+', header=None)[0]

# Remove predictors with too many na's
max_na_pred = 20
index_many_na = np.where(secom.isnull().sum(axis=0) > max_na_pred)[0]
secom = secom.drop(index_many_na, axis=1) 
print('{:d} features were dropped due to a large number of NA\'s.'.format(index_many_na.size))

# Remove incomplete cases
index_drop = np.where(secom.isnull().any(axis = 1))[0]
secom = secom.drop(index_drop, axis=0)
y = y.drop(index_drop)

X = secom.as_matrix()
print('Removing additional {:d} features for identifiability.'.format(np.sum(np.var(X, 0) == 0)))
X = X[:, np.var(X, 0) > 0]
X = (X - np.mean(X, 0)) / np.std(X, 0)
X = np.hstack((np.ones((X.shape[0], 1)), X)) # Intercept
y = y.as_matrix().astype('float')

n_param = X.shape[1]
n_disc = n_param # None of the conditional densities is smooth.


### Define functions to compute the posterior.
def f(theta, req_grad=True):
    """
    Computes the log posterior density and its gradient. 
    
    Params:
    ------
    theta : ndarray
    req_grad : bool
        If True, returns the gradient along with the log density.
    
    Returns:
    -------
    logp : float
    grad : ndarray
    aux : Any
        Any computed quantities that can be re-used by the 
        subsequent calls to the function 'f_updated' to save on
        computation.
    """
    
    logp = 0
    grad = np.zeros(len(y))
    
    # Contribution from the prior.
    logp += - np.sum(theta ** 2) / 2
    
    # Contribution from the likelihood.
    y_hat = np.dot(X, theta)
    loglik = np.count_nonzero(y * y_hat > 0)
    logp += loglik
    
    aux = (loglik, y_hat)
    return logp, np.zeros(len(theta)), aux

def f_update(theta, dtheta, j, aux):
    """
    Computes the difference in the log conditional density 
    along a given parameter index 'j'.
    
    Params:
    ------
    theta : ndarray
    dtheta : float
        Amount by which the j-th parameter is updated.
    j : int
        Index of the parameter to update.
    aux : Any
        Computed quantities from the most recent call to functions
        'f' or 'f_update' that can be re-used to save on computation.
    
    Returns:
    -------
    logp_diff : float
    aux_new : Any
    """
    
    loglik_prev, y_hat = aux
    y_hat = y_hat + X[:,j] * dtheta
    
    logp_diff = (theta[j] ** 2 - (theta[j] + dtheta) ** 2) / 2
    
    # Contribution from the likelihood.
    loglik = np.count_nonzero(y * y_hat > 0)
    logp_diff += loglik - loglik_prev
    
    aux_new = (loglik, y_hat)
    return logp_diff, aux_new