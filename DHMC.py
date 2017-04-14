import numpy as np
import math
import time

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

    def pwc_laplace_leapfrog(self, f, f_update, dt, theta0, p0, grad, n_disc=0, M=None):
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
            theta[:-n_disc] = theta[:-n_disc] + dt / 2 * p[:-n_disc] / M[:-n_disc]

            # Update discrete parameters.
            logp, _, aux = f(theta, req_grad=False)
            if math.isinf(logp):
                return theta, p, grad, logp, nfevals
            coord_order = len(theta) - n_disc + np.random.permutation(n_disc)
            for index in coord_order:
                theta, p, logp = self.update_coordwise(f_update, aux, index, theta, p, M, dt, logp)

            theta[:-n_disc] = theta[:-n_disc] + dt / 2 * p[:-n_disc] / M[:-n_disc]

        logp, grad, _ = f(theta)
        nfevals += 1
        p[:-n_disc] = p[:-n_disc] + 0.5 * dt * grad[:-n_disc]

        return theta, p, grad, logp, nfevals


    def update_coordwise(self, f_update, aux, index, theta, p, M, dt, logp):
        p_sign = math.copysign(1.0, p[index])
        dtheta = p_sign / M[index] * dt
        logp_diff = f_update(theta, dtheta, index, aux)
        dU = - logp_diff
        if abs(p[index]) / M[index] > dU:
            p[index] += - p_sign * M[index] * dU
            theta[index] += dtheta
            logp += logp_diff
        else:
            p[index] = - p[index]
        return theta, p, logp

    ## Proposal scheme: basically identical to the standard HMC except for the integrator and kinetic energy
    def HMC(self, epsilon, n_step, theta0, logp0, grad0):

        p = self.random_momentum()
        joint0 = - self.compute_hamiltonian(logp0, p)

        nfevals_total = 0
        theta, p, grad, logp, nfevals = self.integrator(epsilon, theta0, p, grad0)
        nfevals_total += nfevals
        for i in range(1, n_step):
            if math.isinf(logp):
                break
            theta, p, grad, logp, nfevals = self.integrator(epsilon, theta, p, grad)
            nfevals_total += nfevals

        joint = - self.compute_hamiltonian(logp, p)
        acceptprob = min(1, np.exp(joint - joint0))

        if acceptprob < np.random.rand():
            theta = theta0
            logp = logp0
            grad = grad0

        return theta, logp, grad, acceptprob, nfevals_total

    # Integrator and kinetic energy functions for the proposal scheme.
    def integrator(self, dt, theta0, p0, grad):
        return self.pwc_laplace_leapfrog(self.f, self.f_update, dt, theta0, p0, grad, self.n_disc, self.M)

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
        logp, grad, _ = self.f(theta)
        for i in range(n_sample + n_burnin):
            dt = np.random.uniform(dt_range[0], dt_range[1])
            nstep = np.random.randint(nstep_range[0], nstep_range[1] + 1)
            theta, logp, grad, accept_prob[i], nfevals = self.HMC(dt, nstep, theta, logp, grad)
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