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
Curve data classes.
"""

import dataclasses
from typing import Any, Dict, Callable, Union, List, Tuple, Optional, Iterable
import numpy as np

from qiskit_experiments.framework import FitVal
from qiskit_experiments.exceptions import AnalysisError


@dataclasses.dataclass(frozen=True)
class SeriesDef:
    """Description of curve."""

    # Arbitrary callback to define the fit function. First argument should be x.
    fit_func: Callable

    # Keyword dictionary to define the series with circuit metadata
    filter_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # Name of this series. This name will appear in the figure and raw x-y value report.
    name: str = "Series-0"

    # Color of this line.
    plot_color: str = "black"

    # Symbol to represent data points of this line.
    plot_symbol: str = "o"

    # Whether to plot fit uncertainty for this line.
    plot_fit_uncertainty: bool = False

    # Latex description of this fit model
    model_description: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class CurveData:
    """Set of extracted experiment data."""

    # Name of this data set
    label: str

    # X data
    x: np.ndarray

    # Y data (measured data)
    y: np.ndarray

    # Error bar
    y_err: np.ndarray

    # Maping of data index to series index
    data_index: Union[np.ndarray, int]

    # Metadata associated with each data point. Generated from the circuit metadata.
    metadata: np.ndarray = None


@dataclasses.dataclass(frozen=True)
class FitData:
    """Set of data generated by the fit function."""

    # Order sensitive fit parameter values
    popt: np.ndarray

    # Order sensitive parameter name list
    popt_keys: List[str]

    # Order sensitive fit parameter uncertainty
    popt_err: np.ndarray

    # Covariance matrix
    pcov: np.ndarray

    # Reduced Chi-squared value of fit curve
    reduced_chisq: float

    # Degree of freedom
    dof: int

    # X data range
    x_range: Tuple[float, float]

    # Y data range
    y_range: Tuple[float, float]

    def fitval(self, key: str, unit: Optional[str] = None) -> FitVal:
        """A helper method to get fit value object from parameter key name.

        Args:
            key: Name of parameters to extract.
            unit: Optional. Unit of this value.

        Returns:
            FitVal object.

        Raises:
            ValueError: When specified parameter is not defined.
        """
        try:
            index = self.popt_keys.index(key)
            return FitVal(
                value=self.popt[index],
                stderr=self.popt_err[index],
                unit=unit,
            )
        except ValueError as ex:
            raise ValueError(f"Parameter {key} is not defined.") from ex


@dataclasses.dataclass
class ParameterRepr:
    """Detailed description of fitting parameter."""

    # Fitter argument name
    name: str

    # Unicode representation
    repr: Optional[str] = None

    # Unit
    unit: Optional[str] = None


# pylint: disable=invalid-name
class FitOptions:
    """Collection of fitting options.

    This class is initialized with a list of parameter names,
    and automatically format given initial parameters and boundaries based on it.

    This class provides ``__hash__`` and ``__eq__`` methods to evaluate duplication.
    """

    def __init__(self, parameters: List[str]):
        """Create a new fit options."""

        # no direct access to members for safety hash. these are usually mutable objects.
        self.__p0 = {p: None for p in parameters}
        self.__bounds = {p: (-np.inf, np.inf) for p in parameters}
        self.__extra_opts = dict()

    @property
    def p0(self) -> Dict[str, float]:
        """Return initial guesses."""
        return self.__p0.copy()

    @p0.setter
    def p0(self, new_p0: Union[Dict[str, float], Iterable[float]]):
        """Set new initial guesses.

        Raises:
            AnalysisError: New value is array-like but number of element doesn't match.
        """
        if new_p0 is None:
            return

        if not isinstance(new_p0, dict):
            # format to dictionary
            if len(new_p0) != len(self.__p0):
                raise AnalysisError(
                    "Initial guess is provided as an array with invalid length. "
                    f"{len(self.__p0)} parameters should be provided."
                )
            new_p0 = dict(zip(self.__p0.keys(), new_p0))

        # update initial guesses
        for k, v in new_p0.items():
            if k in self.__p0 and v is not None:
                self.__p0[k] = float(v)

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return boundaries."""
        return self.__bounds.copy()

    @bounds.setter
    def bounds(self, new_bounds: Union[Iterable[Tuple], Dict[str, Tuple]]):
        """Set new boundaries.

        Raises:
            AnalysisError: New value is array-like but number of element doesn't match.
            AnalysisError: One of new value is not a tuple of min max value.
        """
        if new_bounds is None:
            return

        if not isinstance(new_bounds, dict):
            # format to dictionary
            if len(new_bounds) != len(self.__bounds):
                raise AnalysisError(
                    "Boundary is provided as an array with invalid length. "
                    f"{len(self.__bounds)} boundaries should be provided."
                )
            new_bounds = dict(zip(self.__bounds.keys(), new_bounds))

        # update bounds
        for k, v in new_bounds.items():
            if k in self.__bounds and v is not None:
                try:
                    minv, maxv = v
                except (TypeError, ValueError) as ex:
                    raise AnalysisError(
                        f"Boundary of {k} is not a tuple of min-max values."
                    ) from ex
                self.__bounds[k] = (float(minv), float(maxv))

    @property
    def extra_opts(self):
        """Returns extra options provided to the fitter."""
        return self.__extra_opts.copy()

    @extra_opts.setter
    def extra_opts(self, new_options: Dict[str, Any]):
        """Set extra options provided to the fitter."""
        if new_options is None:
            return

        self.__extra_opts.update(**new_options)

    @property
    def options(self) -> Dict[str, Any]:
        """Generate full argument for the fitter."""
        return {"p0": self.p0, "bounds": self.bounds, **self.extra_opts}

    def __hash__(self):
        return hash(
            (
                tuple(sorted(self.__p0.items())),
                tuple(sorted(self.__bounds.items())),
                tuple(sorted(self.__extra_opts.items())),
            )
        )

    def __eq__(self, other):
        if isinstance(other, FitOptions):
            return hash(self) == hash(other)
        return False
