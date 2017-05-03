import numpy as np
import math
import time


# Utility function to check that the computed gradient value is correct.
def test_grad(f, x_center, x_sd=None, rel_tol=.01, dx=10 ** -6, n_test=10, msg=True):
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
        [x_center + x_sd * np.random.normal(size=len(x_center))
            for n in range(n_test)]

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

        test_pass = np.allclose(grad, grad_est, rtol=rel_tol,
                                atol=abs_tol)
        if not test_pass:
            if msg:
                print(
                    'Test failed: the computed gradient does not match with the centered \n' \
                    + 'difference approximation to the tolerance level.')
            break

    if test_pass and msg:
        print('Test passed! The computed gradient seems to be correct.')

    return test_pass, x, grad, grad_est



class DHMCSampler(object):

    def __init__(self, f, f_update, n_disc, n_param, scale=None):
        if scale is None:
            scale = np.ones(n_param)
        self.M = 1 / np.concatenate((scale[:-n_disc] ** 2, scale[-n_disc:]))  # Set the scale of p to be inversely
        # proportional to the scale of theta. 1 / [var(p[0]), std(p[1])]
        self.n_param = n_param
        self.n_disc = n_disc
        self.f = f
        self.f_update = f_update

    # Utility function to check that the returned values of f and f_updates are
    # all consistent.

    def test_cont_grad(self, theta0, sd, rtol=.01, dx=10**-6, n_test=10):
        # Wrapper function for test_grad. Checks that the gradient with respect
        # to the continuous parameters.

        for i in range(n_test):
            theta = theta0.copy()
            theta[-self.n_disc:] += sd * np.random.randn(self.n_disc)
            def f_test(theta_cont):
                logp, grad, aux = self.f(np.concatenate((theta_cont, theta0[-self.n_disc:])))
                grad = grad[:-self.n_disc]
                return logp, grad

            test_pass, theta_cont, grad, grad_est = \
                test_grad(f_test, theta0[:-self.n_disc], sd, rtol, dx, n_test=1, msg=False)

            if not test_pass:
                print('Test failed: the computed gradient does not match with the centered \n' \
                        + 'difference approximation to the tolerance level.')
                break

        if test_pass:
            print('Test passed! The computed gradient seems to be correct.')

        return test_pass, theta_cont, grad, grad_est

    def test_update(self, theta0, sd, n_test=10, atol=10 ** -3, rtol=10 ** -3):

        test_pass = True
        for i in range(n_test):
            index = np.random.randint(self.n_param - self.n_disc, self.n_param)
            theta = theta0 + .1 * sd * np.random.randn(len(theta0))
            dtheta = sd * np.random.randn(1)
            theta_new = theta.copy()
            theta_new[index] += dtheta
            logp_prev, _, aux = self.f(theta)
            logp_curr, _, _ = self.f(theta_new)
            logp_diff, _ = self.f_update(theta, dtheta, index, aux)
            both_inf = math.isinf(logp_diff) and \
                       math.isinf(logp_curr - logp_prev)
            if not both_inf:
                abs_err = abs(logp_diff - (logp_curr - logp_prev))
                if logp_curr == logp_prev:
                    rel_err = 0
                else:
                    rel_err = abs_err / abs(logp_curr - logp_prev)
                if abs_err > atol or rel_err > rtol:
                    test_pass = False
                    break

        if test_pass:
            print('Test passed! The logp differences agree.')
        else:
            print('Test failed: the logp differences do not agree.')

        return test_pass, theta, logp_curr - logp_prev, logp_diff

    def pwc_laplace_leapfrog(self, f, f_update, dt, theta0, p0, logp, grad, aux, n_disc=0, M=None):
        # Params
        # ------
        # f: function(theta, req_grad)
        #   Returns the log probability and, if req_grad is True, its gradient.
        #   The gradient for discrete parameters should be zero.
        # f_update: function(theta, dtheta, index, aux)
        #   Computes the difference in the log probability when theta[index] is
        #   modified by dtheta. The input 'aux' is whatever the quantity saved from
        #   the previous call to 'f' that can be recycled.
        # M: column vector representing diagonal mass matrix
        # n_disc: the number of discrete parameters. The parameters theta[:-n_disc] are assumed continuous.

        if M is None:
            M = np.ones(len(theta0))

        # Flip the direction of p if necessary so that the rest of code can assume that dt > 0.
        theta = theta0.copy()
        p = p0.copy()

        nfevals = 0
        p[:-n_disc] = p[:-n_disc] + 0.5 * dt * grad[:-n_disc]
        if n_disc == 0:
            theta = theta + dt * p / M
        else:
            # Update continuous parameters if any.
            if self.n_param != self.n_disc:
                theta[:-n_disc] = theta[:-n_disc] + dt / 2 * p[:-n_disc] / M[:-n_disc]
                logp, _, aux = f(theta, req_grad=False)

            # Update discrete parameters.
            if math.isinf(logp):
                return theta, p, grad, logp, nfevals
            coord_order = len(theta) - n_disc + np.random.permutation(n_disc)
            for index in coord_order:
                theta, p, logp, aux = self.update_coordwise(f_update, aux, index, theta, p, M, dt, logp)

            theta[:-n_disc] = theta[:-n_disc] + dt / 2 * p[:-n_disc] / M[:-n_disc]

        if self.n_param != self.n_disc:
            logp, grad, aux = f(theta)
            nfevals += 1
            p[:-n_disc] = p[:-n_disc] + 0.5 * dt * grad[:-n_disc]

        return theta, p, grad, logp, aux, nfevals

    def update_coordwise(self, f_update, aux, index, theta, p, M, dt, logp):
        p_sign = math.copysign(1.0, p[index])
        dtheta = p_sign / M[index] * dt
        logp_diff, aux_new = f_update(theta, dtheta, index, aux)
        dU = - logp_diff
        if abs(p[index]) / M[index] > dU:
            p[index] += - p_sign * M[index] * dU
            theta[index] += dtheta
            logp += logp_diff
            aux = aux_new
        else:
            p[index] = - p[index]
        return theta, p, logp, aux

    ## Proposal scheme: basically identical to the standard HMC except for the integrator and kinetic energy
    def HMC(self, epsilon, n_step, theta0, logp0, grad0, aux0):

        p = self.random_momentum()
        joint0 = - self.compute_hamiltonian(logp0, p)

        nfevals_total = 0
        theta, p, grad, logp, aux, nfevals = self.integrator(epsilon, theta0, p, logp0, grad0, aux0)
        nfevals_total += nfevals
        for i in range(1, n_step):
            if math.isinf(logp):
                break
            theta, p, grad, logp, aux, nfevals = self.integrator(epsilon, theta, p, logp, grad, aux)
            nfevals_total += nfevals

        joint = - self.compute_hamiltonian(logp, p)
        acceptprob = min(1, np.exp(joint - joint0))

        if acceptprob < np.random.rand():
            theta = theta0
            logp = logp0
            grad = grad0

        return theta, logp, grad, aux, acceptprob, nfevals_total

    # Integrator and kinetic energy functions for the proposal scheme.
    def integrator(self, dt, theta0, p0, logp, grad, aux):
        return self.pwc_laplace_leapfrog(self.f, self.f_update, dt, theta0, p0, logp, grad, aux, self.n_disc, self.M)

    def random_momentum(self):
        p_cont = np.sqrt(self.M[:-self.n_disc]) * np.random.normal(size = self.n_param - self.n_disc)
        p_disc = self.M[-self.n_disc:] * np.random.laplace(size=self.n_disc)
        return np.concatenate((p_cont, p_disc))

    def compute_hamiltonian(self, logp, p):
        return - logp \
            + np.sum(p[:-self.n_disc] ** 2 / 2 / self.M[:-self.n_disc]) \
            + np.sum(np.abs(p[-self.n_disc:]) / self.M[-self.n_disc:])

    def run_sampler(self, theta0, dt_range, nstep_range, n_burnin, n_sample, seed=None, n_update=10):

        np.random.seed(seed)

        # Run NUTS
        theta = theta0
        n_per_update = math.ceil((n_burnin + n_sample) / n_update)
        nfevals_total = 0
        samples = np.zeros((n_sample + n_burnin, len(theta)))
        logp_samples = np.zeros(n_sample + n_burnin)
        accept_prob = np.zeros(n_sample + n_burnin)

        tic = time.process_time()  # Start clock
        logp, grad, aux = self.f(theta)
        for i in range(n_sample + n_burnin):
            dt = np.random.uniform(dt_range[0], dt_range[1])
            nstep = np.random.randint(nstep_range[0], nstep_range[1] + 1)
            theta, logp, grad, aux, accept_prob[i], nfevals = self.HMC(dt, nstep, theta, logp, grad, aux)
            nfevals_total += nfevals
            samples[i, :] = theta
            logp_samples[i] = logp
            if (i + 1) % n_per_update == 0:
                print('{:d} iterations have been completed.'.format(i + 1))

        toc = time.process_time()
        time_elapsed = toc - tic
        nfevals_per_itr = nfevals_total / (n_sample + n_burnin)
        print('Each iteration required {:.2f} likelihood evaluations on average.'.format(nfevals_per_itr))

        return samples, logp_samples, accept_prob, nfevals_per_itr, time_elapsed