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
Standard RB analysis class.
"""

from typing import List, Dict, Any, Union

import numpy as np
from qiskit.providers import Options

from qiskit_experiments.analysis import (
    CurveAnalysis,
    CurveAnalysisResult,
    SeriesDef,
    CurveData,
    fit_function,
    get_opt_value,
    get_opt_error,
)
from qiskit_experiments.analysis.data_processing import multi_mean_xy_data
from qiskit_experiments.autodocs import (
    OptionsField,
    CurveFitParameter,
    curve_analysis_documentation,
)


@curve_analysis_documentation
class RBAnalysis(CurveAnalysis):
    """Randomized benchmarking analysis."""

    __doc_overview__ = r"""This analysis takes only single series.
This curve is fit by the exponential decay function.
From the fit :math:`\alpha` value this analysis estimates the error per Clifford (EPC)."""

    __doc_equations__ = [r"F(x) = a \alpha^x + b"]

    __doc_fit_params__ = [
        CurveFitParameter(
            name="a",
            description="Height of decay curve.",
            initial_guess=r"Determined by :math:`(y_0 - b) / \alpha^{x_0}`, "
            r"where :math:`b` and :math:`\alpha` are initial guesses.",
            bounds="[0, 1]",
        ),
        CurveFitParameter(
            name="b",
            description="Base line.",
            initial_guess=r"Determined by :math:`(1/2)^n` where :math:`n` is the number of qubit.",
            bounds="[0, 1]",
        ),
        CurveFitParameter(
            name=r"\alpha",
            description="Depolarizing parameter. This is the fit parameter of main interest.",
            initial_guess=r"Determined by the slope of :math:`(y - b)^{-x}` of the first and "
            "the second data point.",
            bounds="[0, 1]",
        ),
    ]

    __series__ = [
        SeriesDef(
            fit_func=lambda x, a, alpha, b: fit_function.exponential_decay(
                x, amp=a, lamb=-1.0, base=alpha, baseline=b
            ),
            plot_color="blue",
        )
    ]

    @classmethod
    def _default_options(cls) -> Union[Options, Dict[str, OptionsField]]:
        """Return default options."""
        default_options = super()._default_options()

        # update default values
        default_options["p0"].default = {"a": None, "alpha": None, "b": None}
        default_options["bounds"].default = {"a": (0.0, 1.0), "alpha": (0.0, 1.0), "b": (0.0, 1.0)}
        default_options["xlabel"].default = "Clifford Length"
        default_options["ylabel"].default = "P(0)"
        default_options["fit_reports"].default = {"alpha": "\u03B1", "EPC": "EPC"}

        return default_options

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Fitter options."""
        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")

        curve_data = self._data()
        initial_guess = self._initial_guess(curve_data.x, curve_data.y, self._num_qubits)
        fit_option = {
            "p0": {
                "a": user_p0["a"] or initial_guess["a"],
                "alpha": user_p0["alpha"] or initial_guess["alpha"],
                "b": user_p0["b"] or initial_guess["b"],
            },
            "bounds": {
                "a": user_bounds["a"] or (0.0, 1.0),
                "alpha": user_bounds["alpha"] or (0.0, 1.0),
                "b": user_bounds["b"] or (0.0, 1.0),
            },
        }
        fit_option.update(options)

        return fit_option

    @staticmethod
    def _initial_guess(
        x_values: np.ndarray, y_values: np.ndarray, num_qubits: int
    ) -> Dict[str, float]:
        """Create initial guess with experiment data."""
        fit_guess = {"a": 0.95, "alpha": 0.99, "b": 1 / 2 ** num_qubits}

        # Use the first two points to guess the decay param
        dcliff = x_values[1] - x_values[0]
        dy = (y_values[1] - fit_guess["b"]) / (y_values[0] - fit_guess["b"])
        alpha_guess = dy ** (1 / dcliff)

        if alpha_guess < 1.0:
            fit_guess["alpha"] = alpha_guess

        if y_values[0] > fit_guess["b"]:
            fit_guess["a"] = (y_values[0] - fit_guess["b"]) / fit_guess["alpha"] ** x_values[0]

        return fit_guess

    def _format_data(self, data: CurveData) -> CurveData:
        """Take average over the same x values."""
        mean_data_index, mean_x, mean_y, mean_e = multi_mean_xy_data(
            series=data.data_index,
            xdata=data.x,
            ydata=data.y,
            sigma=data.y_err,
            method="sample",
        )
        return CurveData(
            label="fit_ready",
            x=mean_x,
            y=mean_y,
            y_err=mean_e,
            data_index=mean_data_index,
        )

    def _post_analysis(self, analysis_result: CurveAnalysisResult) -> CurveAnalysisResult:
        """Calculate EPC."""
        alpha = get_opt_value(analysis_result, "alpha")
        alpha_err = get_opt_error(analysis_result, "alpha")

        scale = (2 ** self._num_qubits - 1) / (2 ** self._num_qubits)
        analysis_result["EPC"] = scale * (1 - alpha)
        analysis_result["EPC_err"] = scale * alpha_err / alpha

        return analysis_result
