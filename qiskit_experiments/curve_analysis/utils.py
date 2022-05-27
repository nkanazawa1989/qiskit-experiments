# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utils in curve analysis."""

from typing import Union, Optional, List, Dict

import pandas as pd
import numpy as np
from qiskit.utils import detach_prefix
from uncertainties import UFloat, wrap as wrap_function
from lmfit.minimizer import MinimizerResult
from lmfit.model import Model

from qiskit_experiments.curve_analysis.curve_data import CurveFitResult
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import AnalysisResultData


def colors10(index: int) -> str:
    """Return color code for given index.

    Args:
        index: Color index.

    Returns:
        Color code.
    """
    cycle_ind = index % 10
    color_list = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    return color_list[cycle_ind]


def symbols10(index: int) -> str:
    """Return marker symbol code for given index.

    Args:
        index: Marker index.

    Returns:
        Marker symbol code.
    """
    cycle_ind = index % 10
    symbols_list = [
        "o",
        "^",
        "x",
        "v",
        "*",
        "D",
        "s",
        "+",
        "<",
        ">",
    ]
    return symbols_list[cycle_ind]


def is_error_not_significant(
    val: Union[float, UFloat],
    fraction: float = 1.0,
    absolute: Optional[float] = None,
) -> bool:
    """Check if the standard error of given value is not significant.

    Args:
        val: Input value to evaluate. This is assumed to be float or ufloat.
        fraction: Valid fraction of the nominal part to its standard error.
            This function returns ``False`` if the nominal part is
            smaller than the error by this fraction.
        absolute: Use this value as a threshold if given.

    Returns:
        ``True`` if the standard error of given value is not significant.
    """
    if isinstance(val, float):
        return True

    threshold = absolute if absolute is not None else fraction * val.nominal_value
    if np.isnan(val.std_dev) or val.std_dev < threshold:
        return True

    return False


def analysis_result_to_repr(result: AnalysisResultData) -> str:
    """A helper function to create string representation from analysis result data object.

    Args:
        result: Analysis result data.

    Returns:
        String representation of the data.

    Raises:
        AnalysisError: When the result data is not likely fit parameter.
    """
    if not isinstance(result.value, (float, UFloat)):
        raise AnalysisError(f"Result data {result.name} is not a valid fit parameter data type.")

    unit = result.extra.get("unit", None)

    def _format_val(value):
        # Return value with unit with prefix, i.e. 1000 Hz -> 1 kHz.
        if unit:
            try:
                val, val_prefix = detach_prefix(value, decimal=3)
            except ValueError:
                val = value
                val_prefix = ""
            return f"{val: .3g}", f" {val_prefix}{unit}"
        if np.abs(value) < 1e-3 or np.abs(value) > 1e3:
            return f"{value: .4e}", ""
        return f"{value: .4g}", ""

    if isinstance(result.value, float):
        # Only nominal part
        n_repr, n_unit = _format_val(result.value)
        value_repr = n_repr + n_unit
    else:
        # Nominal part
        n_repr, n_unit = _format_val(result.value.nominal_value)

        # Standard error part
        if result.value.std_dev is not None and np.isfinite(result.value.std_dev):
            s_repr, s_unit = _format_val(result.value.std_dev)
            if n_unit == s_unit:
                value_repr = f" {n_repr} \u00B1 {s_repr}{n_unit}"
            else:
                value_repr = f" {n_repr + n_unit} \u00B1 {s_repr + s_unit}"
        else:
            value_repr = n_repr + n_unit

    return f"{result.name} = {value_repr}"


def convert_lmfit_result(
    result: MinimizerResult,
    models: List[Model],
    data: pd.DataFrame,
) -> CurveFitResult:
    """A helper function to convert LMFIT ``MinimizerResult`` into :class:`.CurveFitResult`.

    :class:`.CurveFitResult` is a dataclass that can be serialized with the experiment JSON decoder.
    In addition, this class converts LMFIT ``Parameter`` objects into ufloats so that
    extra parameter computation in the analysis post-processing can perform
    accurate error propagation with parameter correlation.

    Args:
        result: Output from LMFIT ``minimize``.
        models: Model used for the fitting. Function description is extracted.
        data: Data used for the fitting.

    Returns:
        QiskitExperiments :class:`.CurveFitResult` object.
    """
    model_descriptions = {}
    for model in models:
        if hasattr(model, "expr"):
            func_repr = model.expr
        else:
            signature = ", ".join(model.independent_vars + model.param_names)
            func_repr = f"F({signature})"
        model_descriptions[model._name] = func_repr

    if result is None:
        return CurveFitResult(
            model_repr=model_descriptions,
            success=False,
            data=data,
        )

    covar = getattr(result, "covar", None)
    if covar is not None and np.any(np.diag(covar) < 0):
        covar = None

    return CurveFitResult(
        method=result.method,
        model_repr=model_descriptions,
        success=result.success,
        nfev=result.nfev,
        message=result.message,
        dof=result.nfree,
        chisq=result.chisqr,
        reduced_chisq=result.redchi,
        aic=result.aic,
        bic=result.bic,
        params={name: param.value for name, param in result.params.items()},
        var_names=result.var_names,
        data=data,
        covar=covar,
    )


def eval_with_uncertainties(
    x: np.ndarray,
    model: Model,
    params: Dict[str, UFloat],
) -> np.ndarray:
    """Compute Y values with error propagation.

    Args:
        x: X values.
        model: LMFIT model.
        params: Fitter parameters in correlated ufloats.

    Returns:
        Y values with uncertainty (uarray).
    """
    sub_params = {name: params[name] for name in model.param_names}
    wrapfunc = np.vectorize(wrap_function(model.func))

    return wrapfunc(x=x, **sub_params)
