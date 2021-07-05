# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
A library of parameter guess functions.
"""
# pylint: disable=invalid-name

from typing import Optional, Tuple

import numpy as np
from scipy import signal

from qiskit_experiments.exceptions import AnalysisError


def frequency(x: np.ndarray, y: np.ndarray, method: str = "FFT") -> float:
    """Get frequency of oscillating signal.

    This function provides several approach to estimate signal frequency.
    See Methods section below for details.

    .. note::

        This function returns always positive frequency.

    Args:
        x: Array of x values.
        y: Array of y values.
        method: A method to find signal frequency. See below for details.

    Returns:
        Frequency estimation of oscillation signal.

    Raises:
        AnalysisError: When invalid method is specified.

    Methods
        - ``FFT``: Use numpy fast Fourier transform to find signal frequency.
          This is the standard method to estimate the frequency.
        - ``ACF``: Calculate autocorrelation function with numpy and run scipy peak search.
          Frequency is calculated based on x coordinate of the first peak.
          Because this doesn't pick DC signal, sometimes we can get better estimate
          for noisy low frequency signals.
    """
    if method == "ACF":
        corr = np.correlate(y, y, mode="full")
        corr = corr[corr.size // 2:]
        peak_inds, _ = signal.find_peaks(corr)
        if len(peak_inds) == 0:
            return 0
        return 1 / x[peak_inds[0]]

    if method == "FFT":
        y = y - np.mean(y)
        fft_data = np.fft.fft(y)
        fft_freqs = np.fft.fftfreq(len(x), float(np.mean(np.diff(x))))
        main_freq_arg = np.argmax(np.abs(fft_data))
        f_guess = np.abs(fft_freqs[main_freq_arg])
        return f_guess

    raise AnalysisError(
        f"The specified method {method} is not available in frequency guess function."
    )


def max_height(
        y: np.ndarray,
        percentile: Optional[float] = None,
        absolute: bool = False,
) -> Tuple[float, int]:
    """Get maximum value of y curve and its index.

    Args:
        y: Array of y values.
        percentile: Return that percentile value if provided, otherwise just return max value.
        absolute: Use absolute y value.

    Returns:
        The maximum y value and index.
    """
    if absolute:
        y_ = np.abs(y)
    else:
        y_ = y

    if percentile is not None:
        y_max = np.percentile(y_, percentile)
    else:
        y_max = np.max(y_)

    index = list(y_).index(y_max)

    return y_max, index


def min_height(
        y: np.ndarray,
        percentile: Optional[float] = None,
        absolute: bool = False,
) -> Tuple[float, int]:
    """Get minimum value of y curve and its index.

    Args:
        y: Array of y values.
        percentile: Return that percentile value if provided, otherwise just return min value.
        absolute: Use absolute y value.

    Returns:
        The minimum y value and index.
    """
    if absolute:
        y_ = np.abs(y)
    else:
        y_ = y

    if percentile is not None:
        y_min = np.percentile(y_, percentile)
    else:
        y_min = np.min(y_)

    index = list(y_).index(y_min)

    return y_min, index


def exp_decay(x: np.ndarray, y: np.ndarray) -> float:
    r"""Get exponential decay parameter from monotonically increasing (decreasing) curve.

    This assumes following function form.

    .. math::

        y(x) = e^{\alpha x}

    We can calculate :math:`\alpha` as

    .. math::

        \alpha = \log(y(x)) / x

    To find this number, the numpy polynomial fit with ``deg=1`` is used.

    Args:
        x: Array of x values.
        y: Array of y values.

    Returns:
         Decay rate of signal.
    """
    coeffs = np.polyfit(x, np.log(y), deg=1)

    return float(coeffs[0])


def oscillation_exp_decay(
        x: np.ndarray,
        y: np.ndarray,
        filter_window: int = 5,
        filter_dim: int = 2,
        freq_guess: Optional[float] = None,
) -> float:
    r"""Get exponential decay parameter from oscillating signal.

    This assumes following function form.

    .. math::

        y(x) = e^{\alpha x} F(x),

    where :math:`F(x)` is arbitrary oscillation function oscillating at `freq_guess`.
    This function first applies a Savitzky-Golay filter to y value,
    then run scipy peak search to extract peak positions.
    If `freq_guess` is provided, the search function will be robust to fake peaks due to noise.
    This function calls :py:func:`exp_decay` function against x and y values at peaks.

    Args:
        x: Array of x values.
        y: Array of y values.
        filter_window: Window size of Savitzky-Golay filter. This should be odd number.
        filter_dim: Dimension of Savitzky-Golay filter.
        freq_guess: Optional. Initial frequency guess of :math:`F(x)`.

    Returns:
         Decay rate of signal.
    """
    y_smoothed = signal.savgol_filter(y, window_length=filter_window, polyorder=filter_dim)

    if freq_guess is not None and np.abs(freq_guess) > 0:
        period = 1 / np.abs(freq_guess)
        dt = np.mean(np.diff(x))
        width_samples = int(np.round(0.8 * period / dt))
    else:
        width_samples = 1

    peak_pos, _ = signal.find_peaks(y_smoothed, distance=width_samples)

    if len(peak_pos) < 2:
        return 0.

    x_peaks = x[peak_pos]
    y_peaks = y_smoothed[peak_pos]

    return exp_decay(x_peaks, y_peaks)
