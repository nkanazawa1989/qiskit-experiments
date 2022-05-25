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
Analysis class for curve fitting.
"""
# pylint: disable=invalid-name

import warnings
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
from lmfit.model import Model
from uncertainties import unumpy as unp, UFloat

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import ExperimentData, AnalysisResultData, AnalysisConfig
from qiskit_experiments.warnings import deprecated_function
from .base_curve_analysis import BaseCurveAnalysis, PARAMS_ENTRY_PREFIX
from .curve_data import CurveData
from .utils import analysis_result_to_repr, eval_with_uncertainties


class CurveAnalysis(BaseCurveAnalysis):
    """Base class for curve analysis with single curve group.

    The fit parameters from the series defined under the analysis class are all shared
    and the analysis performs a single multi-objective function optimization.

    See :class:`BaseCurveAnalysis` for overridable method documentation.
    """

    def __init__(self, models: Optional[List[Model]] = None):
        """Initialize data fields that are privately accessed by methods.

        Args:
            models: List of LMFIT ``Model`` class to define fitting functions and
                parameters. If multiple models are provided, the analysis performs
                multi-objective optimization where the parameters with the same name
                are shared among provided models. The model can be initialized with
                the keyword ``data_sort_key`` which is a dictionary to specify the
                circuit metadata that is associated with the model.
                Usually multiple models must be provided with this keyword to
                classify the experiment data into subgroups of fit model.
        """
        super().__init__()

        if hasattr(self, "__fixed_parameters__"):
            warnings.warn(
                "The class attribute __fixed_parameters__ has been deprecated and will be removed. "
                "Now this attribute is absorbed in analysis options as fixed_parameters. "
                "This warning will be dropped in v0.4 along with "
                "the support for the deprecated attribute.",
                DeprecationWarning,
                stacklevel=2,
            )
            # pylint: disable=no-member
            self._options.fixed_parameters = {
                p: self.options.get(p, None) for p in self.__fixed_parameters__
            }

        if hasattr(self, "__series__"):
            warnings.warn(
                "The class attribute __series__ has been deprecated and will be removed. "
                "Now this class attribute is moved to the constructor argument. "
                "This warning will be dropped in v0.5 along with "
                "the support for the deprecated attribute.",
                DeprecationWarning,
                stacklevel=2,
            )
            # pylint: disable=no-member
            models = []
            plot_options = {}
            for series_def in self.__series__:
                models.append(
                    Model(
                        name=series_def.name,
                        func=series_def.fit_func,
                        data_sort_key=series_def.filter_kwargs,
                    )
                )
                plot_options[series_def.name] = {
                    "color": series_def.plot_color,
                    "symbol": series_def.plot_symbol,
                    "canvas": series_def.canvas,
                }
            self.drawer.set_options(plot_options=plot_options)

        self._models = models or []

        #: List[CurveData]: Processed experiment data set. For backward compatibility.
        self.__processed_data_set = {}

    @property
    def parameters(self) -> List[str]:
        """Return parameters of this curve analysis."""
        unite_params = []
        for model in self._models:
            for name in model.param_names:
                if name not in unite_params and name not in self.options.fixed_parameters:
                    unite_params.append(name)
        return unite_params

    # pylint: disable=bad-docstring-quotes
    @deprecated_function(
        last_version="0.4",
        msg=(
            "CurveAnalysis will also drop internal cache of processed data after 0.4. "
            "Relevant method signature has been updated to directly receive curve data "
            "rather than accessing data with this method."
        ),
    )
    def _data(
        self,
        series_name: Optional[str] = None,
        label: Optional[str] = "fit_ready",
    ) -> CurveData:
        """Deprecated. Getter for experiment data set.

        Args:
            series_name: Series name to search for.
            label: Label attached to data set. By default, it returns "fit_ready" data.

        Returns:
            Filtered curve data set.

        Raises:
            AnalysisError: When requested series or label are not defined.
        """
        try:
            data = self.__processed_data_set[label]
        except KeyError as ex:
            raise AnalysisError(f"Requested data with label {label} does not exist.") from ex

        if series_name is None:
            return data
        return data.get_subset_of(series_name)

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["pyplot.Figure"]]:

        # Prepare for fitting
        self._initialize(experiment_data)
        analysis_results = []

        # Run data processing
        processed_data = self._run_data_processing(
            raw_data=experiment_data.data(),
            models=self._models,
        )

        if self.options.plot and self.options.plot_raw_data:
            for model in self._models:
                sub_data = processed_data.get_subset_of(model._name)
                self.drawer.draw_raw_data(
                    x_data=sub_data.x,
                    y_data=sub_data.y,
                    name=model._name,
                )
        # for backward compatibility, will be removed in 0.4.
        self.__processed_data_set["raw_data"] = processed_data

        # Format data
        formatted_data = self._format_data(processed_data)
        if self.options.plot:
            for model in self._models:
                sub_data = formatted_data.get_subset_of(model._name)
                self.drawer.draw_formatted_data(
                    x_data=sub_data.x,
                    y_data=sub_data.y,
                    y_err_data=sub_data.y_err,
                    name=model._name,
                )
        # for backward compatibility, will be removed in 0.4.
        self.__processed_data_set["fit_ready"] = formatted_data

        # Run fitting
        fit_data = self._run_curve_fit(
            curve_data=formatted_data,
            models=self._models,
        )

        if fit_data.success:
            quality = self._evaluate_quality(fit_data)
        else:
            quality = "bad"

        if self.options.return_fit_parameters:
            # Store fit status entry regardless of success.
            # This is sometime useful when debugging the fitting code.
            fit_parameters = AnalysisResultData(
                name=PARAMS_ENTRY_PREFIX + self.__class__.__name__,
                value=fit_data,
                quality=quality,
                extra=self.options.extra,
            )
            analysis_results.append(fit_parameters)

        # Create figure and result data
        if fit_data.success:

            # Create analysis results
            analysis_results.extend(
                self._create_analysis_results(
                    fit_data=fit_data, quality=quality, **self.options.extra.copy()
                )
            )
            # calling old extra entry method for backward compatibility
            if hasattr(self, "_extra_database_entry"):
                warnings.warn(
                    "Method '_extra_database_entry' has been deprecated and will be "
                    "removed after 0.4. Please override new method "
                    "'_create_analysis_results' with updated method signature.",
                    DeprecationWarning,
                )
                deprecated_method = getattr(self, "_extra_database_entry")
                analysis_results.extend(deprecated_method(fit_data))

            # Draw fit curves and report
            if self.options.plot:
                interp_x = np.linspace(*fit_data.x_range, 100)
                for model in self._models:
                    y_data_with_uncertainty = eval_with_uncertainties(
                        x=interp_x,
                        model=model,
                        params=fit_data.ufloat_params,
                    )
                    y_mean = unp.nominal_values(y_data_with_uncertainty)
                    # Draw fit line
                    self.drawer.draw_fit_line(
                        x_data=interp_x,
                        y_data=y_mean,
                        name=model._name,
                    )
                    if fit_data.covar is not None:
                        # Draw confidence intervals with different n_sigma
                        sigmas = unp.std_devs(y_data_with_uncertainty)
                        if np.isfinite(sigmas).all():
                            for n_sigma, alpha in self.drawer.options.plot_sigma:
                                self.drawer.draw_confidence_interval(
                                    x_data=interp_x,
                                    y_ub=y_mean + n_sigma * sigmas,
                                    y_lb=y_mean - n_sigma * sigmas,
                                    name=model._name,
                                    alpha=alpha,
                                )

                # Write fitting report
                report_description = ""
                for res in analysis_results:
                    if isinstance(res.value, (float, UFloat)):
                        report_description += f"{analysis_result_to_repr(res)}\n"
                report_description += r"reduced-$\chi^2$ = " + f"{fit_data.reduced_chisq: .4g}"
                self.drawer.draw_fit_report(description=report_description)

        # Add raw data points
        analysis_results.extend(
            self._create_curve_data(curve_data=formatted_data, models=self._models)
        )

        # Finalize plot
        if self.options.plot:
            self.drawer.format_canvas()
            return analysis_results, [self.drawer.figure]

        return analysis_results, []

    def __getstate__(self):
        state = self.__dict__.copy()
        # Convert models into JSON str.
        # This object includes local function and cannot be pickled.
        source = [m.dumps() for m in state["_models"]]
        state["_models"] = source
        return state

    def __setstate__(self, state):
        model_objs = []
        for source in state.pop("_models"):
            tmp_mod = Model(func=None)
            mod = tmp_mod.loads(s=source)
            model_objs.append(mod)
        self.__dict__.update(state)
        self._models = model_objs

    @classmethod
    def from_config(cls, config: Union[AnalysisConfig, Dict]) -> "CurveAnalysis":
        # For backward compatibility. This will be removed in v0.4.

        instance = super().from_config(config)

        # When fixed param value is hard-coded as options. This is deprecated data structure.
        loaded_opts = instance.options.__dict__

        # pylint: disable=no-member
        deprecated_fixed_params = {
            p: loaded_opts[p] for p in instance.parameters if p in loaded_opts
        }
        if any(deprecated_fixed_params):
            warnings.warn(
                "Fixed parameter value should be defined in options.fixed_parameters as "
                "a dictionary values, rather than a standalone analysis option. "
                "Please re-save this experiment to be loaded after deprecation period. "
                "This warning will be dropped in v0.4 along with "
                "the support for the deprecated fixed parameter options.",
                DeprecationWarning,
                stacklevel=2,
            )
            new_fixed_params = instance.options.fixed_parameters
            new_fixed_params.update(deprecated_fixed_params)
            instance.set_options(fixed_parameters=new_fixed_params)

        return instance
