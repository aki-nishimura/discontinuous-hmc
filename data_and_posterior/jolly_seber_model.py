import numpy as np
import math

# Data (sufficient statistics) from the black-kneed capsid data in Seber (
# 1982).
n = np.array([54, 146, 169, 209, 220, 209, 250, 176, 172, 127, 123, 120, 142])
R = np.array([54, 143, 164, 202, 214, 207, 243, 175, 169, 126, 120, 120])
r = np.array([24, 80, 70, 71, 109, 101, 108, 99, 70, 58, 44, 35])
m = np.array([10, 37, 56, 53, 77, 112, 86, 110, 84, 77, 72, 95]) # Exclude m_1 = 0
z = np.array([14, 57, 71, 89, 121, 110, 132, 121, 107, 88, 60, 0]) # Exclude z_1 = 0
u = np.concatenate(([n[0]], n[1:] - m)).astype('int64')
T = len(n)

# Specify the indices of parameters.
index = {
    "p": np.arange(T),
    "phi": np.arange(T, 2 * T - 1),
    "U": np.arange(2 * T - 1, 3 * T - 1)
}
n_param = 3 * T - 1
n_cont = len(index["phi"]) + len(index["p"])
n_disc = n_param - n_cont

# Hyper-parameters
sigma_B = 500

def pack_param(p, phi, U):
    # Returns a properly transformed and concatenated parameter vector.
    theta = np.zeros(n_param)
    theta[index["phi"]] = np.log(phi / (1 - phi))
    theta[index["p"]] = np.log(p / (1 - p))
    theta[index["U"]] = np.log(U)
    return theta

def unpack_param(theta):
    logit = lambda x: 1 / (1 + np.exp(-x))
    if theta.ndim == 1:
        p = logit(theta[index["p"]])
        phi = logit(theta[index["phi"]])
        U = np.array(np.floor(np.exp(theta[index["U"]])), dtype='int64')
        M = np.zeros(U.shape, dtype='int64')
        M[1] = np.random.binomial(n[0], phi[:, 0])
        for i in range(1, len(index["U"]) - 1):
            M[i + 1] = np.random.binomial(u[i] + M[:, i], phi[:, i])
        N = M + U
    else:
        p = logit(theta[:, index["p"]])
        phi = logit(theta[:, index["phi"]])
        U = np.array(np.floor(np.exp(theta[:, index["U"]])), dtype='int64')
        M = np.zeros(U.shape, dtype='int64')
        M[:, 1] = np.random.binomial(n[0], phi[:, 0])
        for i in range(1, len(index["U"]) - 1):
            M[:, i + 1] = np.random.binomial(u[i] + M[:, i], phi[:, i])
        N = M + U
    return p, phi, U, N

# Pre-compute log factorial to save on computation.
n_max = 5000 # Pre-specified maximum value of the population size
log_factorial = np.cumsum(np.log(np.arange(1, n_max + 1)))
log_factorial = np.insert(log_factorial, 0, 0)

def f(theta, req_grad=True):
    
    # Extract each parameter.
    tilp = theta[index["p"]]
    tilphi = theta[index["phi"]]
    log_U = theta[index["U"]]
    U = np.floor(np.exp(log_U)).astype('int64')
    
    # Check for over-flow condition.
    overflow = np.any(np.abs(tilphi) > 35) or \
        np.any(np.abs(tilp) > 35)
    if overflow:
        return -float('inf'), float('nan'), None
    
    # Tranform the parameters to meaningful scales.
    phi = 1 / (1 + np.exp(-tilphi))
    p = 1 / (1 + np.exp(-tilp))
    
    # Check for the boundary condition.
    outside_bdry = np.any(U < u) or np.any(U > n_max)
    if outside_bdry:
        return -float('inf'), float('nan'), None
    
    grad = np.zeros(len(theta))
    
    # Contributions from the discrete parameter transformations.
    jacobian = np.log((U + 1) / U)
    logp = - np.sum(np.log(jacobian))
    
    if req_grad:
        
        # Contributions from prior.
        U_var = sigma_B ** 2 + phi * (1 - phi) * (U - u)[:-1]
        logp += - np.log(U[0]) + \
            np.sum(- np.log(U_var) / 2 - (U[1:] - phi * (U - u)[:-1]) ** 2 / U_var / 2) + \
            np.sum(np.log(phi * (1 - phi))) + \
            np.sum(np.log(p * (1 - p)))
        dU_var_dphi = (U - u)[:-1] * (1 - 2 * phi)
        grad[index["phi"]] += phi * (1 - phi) * (
            dU_var_dphi * (- 1 / 2 / U_var - (U[1:] - phi * (U - u)[:-1]) ** 2 / U_var ** 2 / 2) + \
            (U[1:] - phi * (U - u)[:-1]) * (U - u)[:-1] / U_var)    
        grad[index["phi"]] += 1 - 2 * phi
        grad[index["p"]] += 1 - 2 * p

        # Contributions from the likelihood of first captures.
        logp += np.sum(
            log_factorial[U] - log_factorial[U - u] + \
            u * tilp + U * np.log(1 - p) 
        )
        grad[index["p"]] += u - U * p

        # Contributions from the likelihood of recaptures.
        chi_grad_p = np.zeros(T)
        chi_grad_phi = np.zeros(T - 1)
        chi_grad_p[T - 1] += - phi[T - 2]
        chi_grad_phi[T - 2] += - p[T - 1]
        chi = 1 - phi[T - 2] * p[T - 1] 
        logp += (R[-1] - r[-1]) * np.log(chi)
        grad[index["p"]] += p * (1 - p) * (R[-1] - r[-1]) / chi * chi_grad_p
        grad[index["phi"]] += phi * (1 - phi) * (R[-1] - r[-1]) / chi * chi_grad_phi
        for i in range(2, T):
            # Compute chi_{T-i-1} and its gradient iteratively using
            # the value of chi_{T-i} and its gradient.
            chi_grad_p *= phi[-i] * (1 - p[-i + 1])
            chi_grad_p[-i + 1] += - phi[-i] * chi
            chi_grad_phi *= phi[-i] * (1 - p[-i + 1])
            chi_grad_phi[-i] += - 1 + (1 - p[-i + 1]) * chi
            chi = 1 - phi[-i] * (1 - (1 - p[-i + 1]) * chi)
            logp += (R[-i] - r[-i]) * np.log(chi)
            grad[index["p"]] += p * (1 - p) * (R[-i] - r[-i]) / chi * chi_grad_p
            grad[index["phi"]] += phi * (1 - phi) * (R[-i] - r[-i]) / chi * chi_grad_phi

        logp += np.sum(
            (z + m) * np.log(phi) + \
            z * np.log(1 - p[1:]) + \
            m * np.log(p[1:])
        ) # Assumes that z[0] = z_2, m[0] = m_2
        grad[index["p"][1:]] += - z * p[1:] + m * (1 - p[1:])
        grad[index["phi"]] += (1 - phi) *(z + m)
        
    else:
        
        # Contributions from prior.
        U_var = sigma_B ** 2 + phi * (1 - phi) * (U - u)[:-1]
        logp += - np.log(U[0]) + \
            np.sum(- np.log(U_var) / 2 - (U[1:] - phi * (U - u)[:-1]) ** 2 / U_var / 2) + \
            np.sum(np.log(phi * (1 - phi))) + \
            np.sum(np.log(p * (1 - p)))

        # Contributions from the likelihood of first captures.
        logp += np.sum(
            log_factorial[U] - log_factorial[U - u] + \
            u * tilp + U * np.log(1 - p) 
        )

        # Contributions from the likelihood of recaptures.
        chi = 1 - phi[T - 2] * p[T - 1] 
        logp += (R[-1] - r[-1]) * np.log(chi)
        for i in range(2, T):
            # Compute chi_{T-i-1} and its gradient iteratively using
            # the value of chi_{T-i} and its gradient.
            chi = 1 - phi[-i] * (1 - (1 - p[-i + 1]) * chi)
            logp += (R[-i] - r[-i]) * np.log(chi)
        logp += np.sum(
            (z + m) * np.log(phi) + \
            z * np.log(1 - p[1:]) + \
            m * np.log(p[1:])
        ) # Assumes that z[0] = z_2, m[0] = m_2
    
    return logp, grad, None

def f_update(theta, dtheta, j, aux):
    # j : index of the parameter to be updated.
    
    i = j - index["U"][0] # Convert to the capture occasion index.
    p = 1 / (1 + math.exp(-theta[index["p"][i]]))
    U = math.floor(math.exp(theta[j]))
    U_prop = math.floor(math.exp(theta[j] + dtheta))
    
    # Check for the boundary condition.
    if U_prop < u[i] or U_prop > n_max:
        return -float('inf'), None
    
    # Contribution from the Jacobian.
    logp_diff = math.log(math.log((U + 1) / U)) - \
        math.log(math.log((U_prop + 1) / U_prop))
        
    # Contributions from the likelihood of first captures.
    logp_diff += log_factorial[U_prop] - log_factorial[U_prop - u[i]] + \
        - (log_factorial[U] - log_factorial[U - u[i]]) + \
        (U_prop - U) * np.log(1 - p) 
        
    # Contributions from the prior. Take advantage of Markovian structure.    
    if i == 0:
        logp_diff += math.log(U / U_prop)
    else:
        phi_prev = 1 / (1 + math.exp(-theta[index["phi"][i - 1]]))
        U_prev = math.floor(math.exp(theta[index["U"][i - 1]]))
        U_prev_var = sigma_B ** 2 + \
            phi_prev * (1 - phi_prev) * (U_prev - u[i - 1])
        logp_diff += 1 / U_prev_var / 2 * (
            - (U_prop - phi_prev * (U_prev - u[i - 1])) ** 2 
            + (U - phi_prev * (U_prev - u[i - 1])) ** 2 
        )
    if i < len(index["U"]) - 1:
        phi = 1 / (1 + math.exp(-theta[index["phi"][i]]))
        U_var = sigma_B ** 2 + phi * (1 - phi) * (U - u[i])
        U_prop_var = sigma_B ** 2 + phi * (1 - phi) * (U_prop - u[i])
        U_next = math.floor(math.exp(theta[index["U"][i + 1]]))
        logp_diff += \
            - (U_next - phi * (U_prop - u[i])) ** 2 / U_prop_var / 2 + \
            (U_next - phi * (U - u[i])) ** 2 / U_var / 2
    
    return logp_diff, None


### Function to conditionally update the discrete parameters
def update_disc(theta):
    # Extract each parameter.

    tilphi = theta[index["phi"]]
    tilp = theta[index["p"]]
    log_U = theta[index["U"]]
    phi = 1 / (1 + np.exp(-tilphi))
    p = 1 / (1 + np.exp(-tilp))
    U = np.floor(np.exp(log_U)).astype('int64')

    for i in np.random.permutation(len(U)):
        U[i] = cond_update_U(phi, p, U, i)

    theta[index["U"]] = np.log(U)
    return theta

def cond_update_U(phi, p, U, i):
    # The possible values of U[i].
    U_i = np.arange(u[i], n_max)

    # Log-likelihood of the 1st capture.
    logp = \
        log_factorial[U_i] - log_factorial[U_i - u[i]] + U_i * np.log(1 - p[i])

    # Contributions from prior. Take advantage of Markovian structure.
    if i == 0:
        logp += - np.log(U_i)
    else:
        U_prev_var = sigma_B ** 2 + \
                     phi[i - 1] * (1 - phi[i - 1]) * (U - u)[i - 1]
        logp += - np.log(U_prev_var) / 2 + \
                - (U_i - phi[i - 1] * (
                U[i - 1] - u[i - 1])) ** 2 / U_prev_var / 2

    if i < len(U) - 1:
        U_i_var = sigma_B ** 2 + \
                  phi[i] * (1 - phi[i]) * (U_i - u[i])
        logp += - np.log(U_i_var) / 2 + \
                - (U[i + 1] - phi[i] * (U_i - u[i])) ** 2 / U_i_var / 2

    # Multi-nomial sampling
    prob = np.exp(logp - np.max(logp))
    prob = prob / np.sum(prob)

    return np.random.choice(U_i, size=1, p=prob)