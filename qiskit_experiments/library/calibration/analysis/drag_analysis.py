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

"""DRAG pulse calibration experiment."""

from typing import Any, Dict, List, Union
import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.fit_function import cos


class DragCalAnalysis(curve.CurveAnalysis):
    r"""Drag calibration analysis based on a fit to a cosine function.

    # section: fit_model

        Analyse a Drag calibration experiment by fitting three series each to a cosine function.
        The three functions share the phase parameter (i.e. beta) but each have their own amplitude,
        baseline, and frequency parameters (which therefore depend on the number of repetitions of
        xp-xm). Several initial guesses are tried if the user does not provide one.

        .. math::

            y = {\rm amp} \cos\left(2 \pi\cdot {\rm freq}_i\cdot x - 2 \pi \beta\right) + {\rm base}

    # section: fit_parameters
        defpar \rm amp:
            desc: Amplitude of all series.
            init_guess: The maximum y value less the minimum y value. 0.5 is also tried.
            bounds: [-2, 2] scaled to the maximum signal value.

        defpar \rm base:
            desc: Base line of all series.
            init_guess: The average of the data. 0.5 is also tried.
            bounds: [-1, 1] scaled to the maximum signal value.

        defpar {\rm freq}_i:
            desc: Frequency of the :math:`i` th oscillation.
            init_guess: The frequency with the highest power spectral density.
            bounds: [0, inf].

        defpar \beta:
            desc: Common beta offset. This is the parameter of interest.
            init_guess: Linearly spaced between the maximum and minimum scanned beta.
            bounds: [-min scan range, max scan range].
    """

    __series__ = [
        curve.SeriesDef(
            fit_func=lambda x, amp, freq0, freq1, freq2, beta, base: cos(
                x, amp=amp, freq=freq0, phase=-2 * np.pi * freq0 * beta, baseline=base
            ),
            plot_color="blue",
            name="series-0",
            filter_kwargs={"series": 0},
            plot_symbol="o",
            model_description=r"{\rm amp} \cos\left(2 \pi\cdot {\rm freq}_0\cdot x "
            r"- 2 \pi \beta\right) + {\rm base}",
        ),
        curve.SeriesDef(
            fit_func=lambda x, amp, freq0, freq1, freq2, beta, base: cos(
                x, amp=amp, freq=freq1, phase=-2 * np.pi * freq1 * beta, baseline=base
            ),
            plot_color="green",
            name="series-1",
            filter_kwargs={"series": 1},
            plot_symbol="^",
            model_description=r"{\rm amp} \cos\left(2 \pi\cdot {\rm freq}_1\cdot x "
            r"- 2 \pi \beta\right) + {\rm base}",
        ),
        curve.SeriesDef(
            fit_func=lambda x, amp, freq0, freq1, freq2, beta, base: cos(
                x, amp=amp, freq=freq2, phase=-2 * np.pi * freq2 * beta, baseline=base
            ),
            plot_color="red",
            name="series-2",
            filter_kwargs={"series": 2},
            plot_symbol="v",
            model_description=r"{\rm amp} \cos\left(2 \pi\cdot {\rm freq}_2\cdot x "
            r"- 2 \pi \beta\right) + {\rm base}",
        ),
    ]

    @classmethod
    def _default_options(cls):
        """Return the default analysis options.

        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.result_parameters = ["beta"]
        default_options.xlabel = "Beta"
        default_options.ylabel = "Signal (arb. units)"

        return default_options

    def _generate_fit_guesses(
            self, opt: curve.FitOptions
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Compute the initial guesses.

        Args:
            opt: Fit options filled with user provided guess and bounds.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        # Use a fast Fourier transform to guess the frequency.
        x_data = self._data("series-0").x
        min_beta, max_beta = min(x_data), max(x_data)

        for i in range(3):
            curve_data = self._data(f"series-{i}")
            opt.p0[f"freq{i}"] = curve.guess.frequency(curve_data.x, curve_data.y)

        max_abs_y, _ = curve.guess.max_height(self._data().y, absolute=True)
        freq_bound = max(10 / opt.p0["freq0"], max(x_data))

        opt.bounds["amp"] = -2 * max_abs_y, 2 * max_abs_y
        opt.bounds["freq0"] = 0, np.inf
        opt.bounds["freq1"] = 0, np.inf
        opt.bounds["freq2"] = 0, np.inf
        opt.bounds["beta"] = -freq_bound, freq_bound
        opt.bounds["base"] = -max_abs_y, max_abs_y

        opt.p0["amp"] = 0.5
        opt.p0["base"] = 0.5

        # Drag curves can sometimes be very flat, i.e. averages of y-data
        # and min-max do not always make good initial guesses. We therefore add
        # 0.5 to the initial guesses.
        options = []
        for beta_guess in np.linspace(min_beta, max_beta, 20):
            new_opt = opt.copy()
            new_opt.p0["beta"] = beta_guess
            options.append(new_opt)

        return options

    def _evaluate_quality(self, fit_data: curve.FitData) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three,
            - a DRAG parameter value within the first period of the lowest number of repetitions,
            - an error on the drag beta smaller than the beta.
        """
        fit_beta = fit_data.fitval("beta").value
        fit_beta_err = fit_data.fitval("beta").stderr
        fit_freq0 = fit_data.fitval("freq0").value

        criteria = [
            fit_data.reduced_chisq < 3,
            fit_beta < 1 / fit_freq0,
            fit_beta_err < abs(fit_beta),
        ]

        if all(criteria):
            return "good"

        return "bad"
