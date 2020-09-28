from __future__ import division
from __future__ import absolute_import
import numpy as np


def make_fourier_vdm(n, inverse):
    i = np.arange(n, dtype=np.float64)
    imat = i[:, np.newaxis]*i/n
    result = np.exp((2j*np.pi)*imat)

    if inverse:
        result = result.T.conj()/n
    return result


def make_fourier_mode_extender(m, n, dtype):
    k = min(m, n)
    result = np.zeros((m, n), dtype)

    # http://docs.scipy.org/doc/numpy/reference/routines.fft.html
    if k % 2 == 0:
        peak_pos_freq = k/2
    else:
        peak_pos_freq = (k-1)/2

    num_pos_freq = peak_pos_freq + 1
    num_neg_freq = k-num_pos_freq

    eye = np.eye(k)
    result[:num_pos_freq, :num_pos_freq] = eye[:num_pos_freq, :num_pos_freq]
    result[-num_neg_freq:, -num_neg_freq:] = eye[-num_neg_freq:, -num_neg_freq:]
    return result


def make_fourier_interp_matrix(m, n):
    return np.dot(
            np.dot(
                make_fourier_vdm(m, inverse=False),
                make_fourier_mode_extender(m, n, np.float64)),
            make_fourier_vdm(n, inverse=True))
