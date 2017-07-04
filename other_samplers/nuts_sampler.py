import numpy as np
import math
import time # For benchmarking.

# TODO: support a non-identity matrix.

# Assign the default integrator for NUTS.

def leapfrog(f, epsilon, theta, p, grad):

    p = p + 0.5 * epsilon * grad
    theta = theta + epsilon * p
    logp, grad = f(theta)
    p = p + 0.5 * epsilon * grad
    nfevals = 1

    return theta, p, grad, logp, nfevals

def newtonian_energy(logp, p):
    return - logp + 0.5 * np.dot(p, p)

def newtonian_momentum(d):
    return np.random.randn(d)


# The integrator and momentum distribution can be adjusted and NUTS remains
# a valid reversible proposal mechanism.
integrator = leapfrog
compute_hamiltonian = newtonian_energy
random_momentum = newtonian_momentum

# Utility function to check that the computed gradient value is correct.
def test_grad(f, x_center, x_sd=None, rel_tol=.1, dx=10 ** -6, n_test=10):
    # Compare the computed gradient to a finite difference approximation
    # at 'n_test' randomly generated points.
    # Params:
    # f : function that returns a real-value and gradient.
    # x_center : vector specifying the center of randomly generated points.
    # x_var : vector specifying the variance of randomly generated points.

    x_center = np.array(x_center, ndmin=1)
    if x_sd is None:
        x_sd = np.ones(len(x_center))
    x_sd = np.array(x_sd, ndmin=1)

    abs_tol = dx
    x_test = \
        [x_center + x_sd * np.random.normal(size=len(x_center)) for n in
         range(n_test)]

    for x in x_test:
        grad_est = np.zeros(len(x_center))
        for i in range(len(x_center)):
            x_minus = x.copy()
            x_minus[i] -= dx
            x_plus = x.copy()
            x_plus[i] += dx

            f_minus, _ = f(x_minus)
            f_plus, _ = f(x_plus)
            grad_est[i] = (f_plus - f_minus) / (2 * dx)

        _, grad = f(x)

        test_pass = np.allclose(grad, grad_est, rtol=rel_tol, atol=abs_tol)
        if not test_pass:
            print(
            'The computed gradient does not match with the finite centered \n' \
            + 'difference approximation to the tolerance level.')
            break

    if test_pass:
        print('The computed gradient seems to be correct.')

    return test_pass, x, grad, grad_est

# Main NUTS codes.

def nuts(f, epsilon, theta, logp, grad, max_depth=10, warnings=True):

    d = len(theta)

    # Resample momenta.
    p = random_momentum(d)

    # joint lnp of theta and momentum r
    joint = - compute_hamiltonian(logp, p)

    # Resample u ~ uniform([0, exp(joint)]).
    # Equivalent to (log(u) - joint) ~ exponential(1).
    logu = joint - np.random.exponential()

    # initialize the tree
    thetaminus = theta
    thetaplus = theta
    pminus = p
    pplus = p
    gradminus = grad
    gradplus = grad

    nfevals_total = 0
    depth = 0
    n = 1  # Initially the only valid point is the initial point.
    stop = False
    while not stop:
        # Choose a direction. -1 = backwards, 1 = forwards.
        dir = int(2 * (np.random.uniform() < 0.5) - 1)

        # Double the size of the tree.
        if (dir == -1):
            thetaminus, pminus, gradminus, _, _, _, thetaprime, gradprime, logpprime, nprime, stopprime, alpha, nalpha, nfevals = \
                build_tree(thetaminus, pminus, gradminus, logu, dir, depth, epsilon, f, joint)
        else:
            _, _, _, thetaplus, pplus, gradplus, thetaprime, gradprime, logpprime, nprime, stopprime, alpha, nalpha, nfevals = \
                build_tree(thetaplus, pplus, gradplus, logu, dir, depth, epsilon, f, joint)
        nfevals_total += nfevals

        # Use Metropolis-Hastings to decide whether or not to move to a
        # point from the half-tree we just generated.
        if (not stopprime) and (np.random.uniform() < nprime / n):
            theta = thetaprime
            logp = logpprime
            grad = gradprime
        # Update number of valid points we've seen.
        n += nprime
        # Decide if it's time to stop.
        stop = stopprime or stop_criterion(thetaminus, thetaplus, pminus, pplus)
        # Increment depth.
        depth += 1
        if depth >= max_depth:
            stop = True
            if warnings:
                print('The max depth of {:d} has been reached.'.format(max_depth))

    alpha_ave = alpha / nalpha

    return theta, logp, grad, alpha_ave, nfevals_total

def stop_criterion(thetaminus, thetaplus, pminus, pplus):
    """ Check for the U-turn condition.
    INPUTS
    ------
    thetaminus, thetaplus: ndarray[float, ndim=1]
        under and above position
    pminus, pplus: ndarray[float, ndim=1]
        under and above momentum
    """
    dtheta = thetaplus - thetaminus
    return (np.dot(dtheta, pminus) < 0) or (np.dot(dtheta, pplus) < 0)


def build_tree(theta, p, grad, logu, dir, depth, epsilon, f, joint0):
    """The main recursion."""

    nfevals_total = 0
    if (depth == 0):
        # Base case: Take a single leapfrog step in the direction dir.
        thetaprime, pprime, gradprime, logpprime, nfevals = integrator(f, dir * epsilon, theta, p, grad)
        nfevals_total += nfevals
        if math.isinf(logpprime):
            joint = - float('inf')
        else:
            joint = - compute_hamiltonian(logpprime, pprime)
        # Is the new point in the slice?
        nprime = int(logu < joint)
        # Is the simulation wildly inaccurate?
        stopprime = (logu - 100) > joint
        # Set the return values---minus=plus for all things here, since the
        # "tree" is of depth 0.
        thetaminus = thetaprime
        thetaplus = thetaprime
        pminus = pprime
        pplus = pprime
        gradminus = gradprime
        gradplus = gradprime
        # Compute the acceptance probability.
        alphaprime = min(1, np.exp(joint - joint0))
        nalphaprime = 1
    else:
        # Recursion: Implicitly build the height depth-1 left and right subtrees.
        thetaminus, pminus, gradminus, thetaplus, pplus, gradplus, thetaprime, gradprime, logpprime, nprime, stopprime, alphaprime, nalphaprime, nfevals \
             = build_tree(theta, p, grad, logu, dir, depth - 1, epsilon, f, joint0)
        nfevals_total += nfevals
        # No need to keep going if the stopping criteria were met in the first subtree.
        if not stopprime:
            if (dir == -1):
                thetaminus, pminus, gradminus, _, _, _, thetaprime2, gradprime2, logpprime2, nprime2, stopprime2, alphaprime2, nalphaprime2, nfevals \
                    = build_tree(thetaminus, pminus, gradminus, logu, dir, depth - 1, epsilon, f, joint0)
            else:
                _, _, _, thetaplus, pplus, gradplus, thetaprime2, gradprime2, logpprime2, nprime2, stopprime2, alphaprime2, nalphaprime2, nfevals \
                    = build_tree(thetaplus, pplus, gradplus, logu, dir, depth - 1, epsilon, f, joint0)
            nfevals_total += nfevals
            # Choose which subtree to propagate a sample up from.
            if (np.random.uniform() < nprime2 / max(nprime + nprime2, 1)):
                thetaprime = thetaprime2
                gradprime = gradprime2
                logpprime = logpprime2
            # Update the number of valid points.
            nprime = nprime + nprime2
            # Update the stopping criterion.
            stopprime = (stopprime or stopprime2 or stop_criterion(thetaminus, thetaplus, pminus, pplus))
            # Update the acceptance probability statistics.
            alphaprime = alphaprime + alphaprime2
            nalphaprime = nalphaprime + nalphaprime2

    return thetaminus, pminus, gradminus, thetaplus, pplus, gradplus, \
           thetaprime, gradprime, logpprime, nprime, stopprime, \
           alphaprime, nalphaprime, nfevals_total


def nuts_adap(f, n_sample, n_warmup, theta0, delta=0.8, seed=None, epsilon=None, n_update=10):
    """
    Implements the No-U-Turn Sampler (NUTS) of Hoffman & Gelman, 2011.
    Runs n_warmup steps of burn-in, during which it adapts the step size
    parameter epsilon, then starts generating samples to return.
    INPUTS
    ------
    epsilon: float
        step size
    f: callable
        it should return the log probability and gradient evaluated at theta
        logp, grad = f(theta)
    n_mcmc: int
        number of samples to generate.
    n_warmup: int
        the number of steps of burn-in/how long to run the dual averaging
        algorithm to fit the step size epsilon.
    theta0: ndarray[float, ndim=1]
        initial guess of the parameters.
    KEYWORDS
    --------
    delta: float
        targeted average acceptance rate
    OUTPUTS
    -------
    samples: ndarray[float, ndim=2]
    """

    np.random.seed(seed)

    if len(np.shape(theta0)) > 1:
        raise ValueError('theta0 is expected to be a 1-D array')

    if n_warmup > 0:
        theta, epsilon, _, _ = dualAveraging(f, theta0, delta, n_warmup)
        print('The warmup iterations have been completed.')
    else:
        theta = theta0.copy()

    n_per_update = math.ceil(n_sample / n_update)

    nfevals_total = 0
    samples = np.zeros((n_sample, len(theta)))
    logp_samples = np.zeros(n_sample)
    accept_prob = np.zeros(n_sample)

    tic = time.process_time()
    logp, grad = f(theta)
    for i in range(n_sample):
        theta, logp, grad, alpha_ave, nfevals = nuts(f, epsilon, theta, logp, grad)
        nfevals_total += nfevals
        samples[i,:] = theta
        logp_samples[i] = logp
        accept_prob[i] = alpha_ave
        if (i + 1) % n_per_update == 0:
            print('{:d} iterations have been completed.'.format(i+1))
    toc = time.process_time()
    time_elapsed = toc - tic
    nfevals_per_itr = nfevals_total / n_sample

    return samples, logp_samples, epsilon, accept_prob, nfevals_per_itr, time_elapsed

def dualAveraging(f, theta0, delta=.8, n_warmup=500):

    logp, grad = f(theta0)
    if math.isinf(logp):
        raise ValueError('The log density at the initial state was infinity.')

    # Choose a reasonable first epsilon by a simple heuristic.
    epsilon = find_reasonable_epsilon(theta0, grad, logp, f)

    # Parameters to the dual averaging algorithm.
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    mu = math.log(10. * epsilon)

    # Initialize dual averaging algorithm.
    epsilonbar = 1
    epsilon_seq = np.zeros(n_warmup + 1)
    epsilonbar_seq = np.zeros(n_warmup + 1)
    epsilon_seq[0] = epsilon
    epsilonbar_seq[0] = epsilonbar

    Hbar = 0
    theta = theta0
    for i in range(1, n_warmup + 1):
        theta, logp, grad, ave_alpha, _ = \
                nuts(f, epsilon, theta, logp, grad)
        eta = 1 / (i + t0)
        Hbar = (1 - eta) * Hbar + eta * (delta - ave_alpha)
        epsilon = math.exp(mu - math.sqrt(i) / gamma * Hbar)
        eta = i ** -kappa
        epsilonbar = math.exp((1 - eta) * math.log(epsilonbar) + eta * math.log(epsilon))
        epsilon_seq[i] = epsilon
        epsilonbar_seq[i] = epsilonbar

    return theta, epsilonbar, epsilon_seq, epsilonbar_seq


def find_reasonable_epsilon(theta0, grad0, logp0, f):
    """ Heuristic for choosing an initial value of epsilon """

    epsilon = 1.0
    p0 = np.random.normal(0, 1, len(theta0))

    # Figure out what direction we should be moving epsilon.
    _, pprime, gradprime, logpprime, _ = integrator(f, epsilon, theta0, p0, grad0)
    if math.isinf(logpprime):
        acceptprob = 0
    else:
        acceptprob = np.exp(logpprime - logp0 - 0.5 * (np.dot(pprime, pprime) - np.dot(p0, p0)))
    a = 2 * int(acceptprob > 0.5) - 1

    # Keep moving epsilon in that direction until acceptprob crosses 0.5.
    while acceptprob == 0 or ((2 * acceptprob) ** a > 1):
        epsilon = epsilon * (2 ** a)
        _, pprime, _, logpprime, _ = integrator(f, epsilon, theta0, p0, grad0)
        if math.isinf(logpprime):
            acceptprob = 0
            if a == 1: # The last doubling of stepsize was too much.
                epsilon = epsilon / 2
                break
        else:
            acceptprob = np.exp(logpprime - logp0 - 0.5 * (np.dot(pprime, pprime) - np.dot(p0, p0)))

    return epsilon

## HMC codes
def HMC(f, epsilon, n_step, theta0, logp0, grad0):

    p = random_momentum(len(theta0))
    joint0 = - compute_hamiltonian(logp0, p)

    nfevals_total = 0
    theta, p, grad, logp, nfevals = integrator(f, epsilon, theta0, p, grad0)
    nfevals_total += nfevals
    for i in range(1, n_step):
        theta, p, grad, logp, nfevals = integrator(f, epsilon, theta, p, grad)
        nfevals_total += nfevals

    joint = - compute_hamiltonian(logp, p)
    if math.isinf(logp):
        acceptprob = 0
    else:
        acceptprob = min(1, np.exp(joint - joint0))

    if acceptprob < np.random.rand():
        theta = theta0
        logp = logp0
        grad = grad0

    return theta, logp, grad, acceptprob, nfevals_total


