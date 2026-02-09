import numpy as np
from numpy.typing import NDArray


def make_fourier_vdm(n: int, inverse: bool) -> NDArray[np.complex128]:
    i = np.arange(n, dtype=np.float64)
    imat = i[:, np.newaxis]*i/n
    result = np.exp((2j*np.pi)*imat)

    if inverse:
        result = np.conj(result.T)/n
    return result


def make_fourier_mode_extender(m, n, dtype):
    k = min(m, n)
    result = np.zeros((m, n), dtype)

    # https://docs.scipy.org/doc/numpy/reference/routines.fft.html
    if k % 2 == 0:  # noqa: SIM108
        peak_pos_freq = k/2
    else:
        peak_pos_freq = (k-1)/2

    num_pos_freq = peak_pos_freq + 1
    num_neg_freq = k-num_pos_freq

    eye = np.eye(k)
    result[:num_pos_freq, :num_pos_freq] = eye[:num_pos_freq, :num_pos_freq]
    result[-num_neg_freq:, -num_neg_freq:] = eye[-num_neg_freq:, -num_neg_freq:]
    return result


def make_fourier_interp_matrix(m: int, n: int):
    return (make_fourier_vdm(m, inverse=False)
        @ make_fourier_mode_extender(m, n, np.float64)
        @  make_fourier_vdm(n, inverse=True))
