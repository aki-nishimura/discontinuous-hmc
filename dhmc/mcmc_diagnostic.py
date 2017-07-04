import numpy as np
import os # Necessary for coda_ess


def coda_ess(samples, axis=0, normed=False, n_digit=18):
    # Estimates effective sample sizes of samples along the specified axis
    # by calling the R package 'coda' externally. It is a hacky but convenient
    # way to call an R function without having to install rpy2 and
    # its dependencies.

    filenum = np.random.randint(2 ** 31)
    # Append a random number to a file name to avoid conflicts.
    saveto = 'mchain{:d}.csv'.format(filenum)
    loadfrom = 'ess{:d}.csv'.format(filenum)
    if axis == 0:
        np.savetxt(saveto, samples, delimiter=',', fmt='%.{:d}e'.format(n_digit))
    else:
        np.savetxt(saveto, samples.T, delimiter=',', fmt='%.{:d}e'.format(n_digit))

    # Write an R script for computing ESS with the 'coda' package if the script
    # is not already present.
    r_code = "'args <- commandArgs(trailingOnly=T) # Read in the input and output file names\n" \
             + "x <- read.csv(args[1], header=F)\n" \
             + "library(coda)\n" \
             + "ess <- unlist(lapply(x, effectiveSize))\n" \
             + "write.table(ess, args[2], sep=',', row.names=F, col.names=F)'"
    os.system(" ".join(["[[ ! -f compute_coda_ess.R ]] && echo", r_code, ">>", "compute_coda_ess.R"]))

    # Write the data to a text file, read into the R script, and output
    # the result back into a text file.
    os.system(" ".join(["Rscript compute_ess.R", saveto, loadfrom]))
    ess = np.loadtxt(loadfrom, delimiter=',').copy()
    if normed:
        ess = ess / samples.shape[axis]
    os.system(" ".join(["rm -f", saveto, loadfrom]))
    return ess


def batch_ess(samples, n_batch=25, axis=0, normed=False):
    # Estimates effective sample sizes of samples along the specified axis
    # with the method of batch means.
    batch_index = np.linspace(0, samples.shape[axis], n_batch + 1).astype('int')
    batch_list = [np.take(samples, np.arange(batch_index[i], batch_index[i + 1]), axis)
        for i in range(n_batch)]
    batch_mean = np.stack((np.mean(batch, axis) for batch in batch_list), axis)
    mcmc_var = samples.shape[axis] / n_batch * np.var(batch_mean, axis)
    ess = np.var(samples, axis) / mcmc_var
    if not normed: ess *= samples.shape[0]
    return ess


def mono_seq_ess(samples, axis=0, normed=False, mu=None, sigma_sq=None, req_acorr=False):
    # Estimates effective sample sizes of samples along the specified axis
    # with the monotone positive sequence estimator of "Practical Markov
    # Chain Monte Carlo" by Geyer (1992). The estimator is ONLY VALID for
    # reversible Markov chains. The inputs 'mu' and 'sigma_sq' are optional
    # and unnecessary for the most cases in practice.
    #
    # Inputs
    # ------
    # mu, sigma_sq : vectors for the mean E(x) and variance Var(x) if the
    #     analytical (or accurately estimated) value is available. If provided,
    #     it can stabilize the estimate of auto-correlations and hence ESS.
    #     This is intended for research uses when one wants to
    #     accurately quantify the asymptotic efficiency of a MCMC algorithm.
    # req_acorr : bool
    #     If true, a list of estimated auto correlation sequences are returned as the
    #     second output.
    # Returns
    # -------
    # ess : numpy array
    # auto_cor : list of numpy array
    #     auto-correlation estimates of the chain up to the lag beyond which the
    #     auto-correlation can be considered insignificant by the monotonicity
    #     criterion.

    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    if mu is None:
        mu = np.mean(samples, axis)
    if sigma_sq is None:
        sigma_sq = np.var(samples, axis)

    d = samples.shape[1 - axis]
    ess = np.zeros(d)
    auto_cor = []
    for j in range(d):
        if axis == 0:
            x = samples[:, j]
        else:
            x = samples[j, :]
        ess_j, auto_cor_j = ess_1d(x, mu[j], sigma_sq[j])
        ess[j] = ess_j
        auto_cor.append(auto_cor_j)
    if normed:
        ess /= samples.shape[axis]

    if req_acorr:
        return ess, auto_cor
    return ess

def ess_1d(x, mu, sigma_sq):
    n = len(x)
    auto_cor = []

    lag = 0
    auto_cor_sum = 0
    even_auto_cor = compute_acorr(x, lag, mu, sigma_sq)
    auto_cor.append(even_auto_cor)
    auto_cor_sum -= even_auto_cor

    lag += 1
    odd_auto_cor = compute_acorr(x, lag, mu, sigma_sq)
    auto_cor.append(odd_auto_cor)
    running_min = even_auto_cor + odd_auto_cor

    while (even_auto_cor + odd_auto_cor > 0) and (lag + 2 < n):
        running_min = min(running_min, (even_auto_cor + odd_auto_cor))
        auto_cor_sum = auto_cor_sum + 2 * running_min

        lag += 1
        even_auto_cor = compute_acorr(x, lag, mu, sigma_sq)
        auto_cor.append(even_auto_cor)

        lag = lag + 1
        odd_auto_cor = compute_acorr(x, lag, mu, sigma_sq)
        auto_cor.append(odd_auto_cor)

    ess = n / auto_cor_sum
    if auto_cor_sum < 0:  # Rare, but can happen when 'x' shows strong negative correlations.
        ess = float('inf')
    return ess, np.array(auto_cor)


def compute_acorr(x, k, mu, sigma_sq):
    # Returns an estimate of the lag 'k' auto-correlation of a time series 'x'.
    # The estimator is biased towards zero due to the factor (n - k) / n.
    # See Geyer (1992) Section 3.1 and the reference therein for justification.
    n = len(x)
    acorr = (x[:(n - k)] - mu) * (x[k:] - mu)
    acorr = np.mean(acorr) / sigma_sq * (n - k) / n
    return acorr