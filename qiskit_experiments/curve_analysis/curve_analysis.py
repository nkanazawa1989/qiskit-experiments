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

import lmfit
import numpy as np
import pandas as pd
from uncertainties import unumpy as unp

from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.framework import ExperimentData, AnalysisResultData, AnalysisConfig
from .base_curve_analysis import BaseCurveAnalysis
from .curve_data import CurveData, FitOptions, CurveFitResult
from .formatter import shot_weighted_average, sample_average, iwv_average
from .utils import eval_with_uncertainties, convert_lmfit_result


class CurveAnalysis(BaseCurveAnalysis):
    """Base class for curve analysis with single curve group.

    The fit parameters from the series defined under the analysis class are all shared
    and the analysis performs a single multi-objective function optimization.

    A subclass may override these methods to customize the fit workflow.

    .. rubric:: _run_data_processing

    This method performs data processing and returns the processed dataset.
    By default, it internally calls the :class:`DataProcessor` instance from
    the `data_processor` analysis option and processes the experiment data payload
    to create Y data with uncertainty.
    X data and other metadata are generated within this method by inspecting the
    circuit metadata. The series classification is also performed based upon the
    matching of circuit metadata and :attr:`SeriesDef.filter_kwargs`.

    .. rubric:: _format_data

    This method consumes the processed dataset and outputs the formatted dataset.
    By default, this method takes the average of y values over
    the same x values and then sort the entire data by x values.

    .. rubric:: _generate_fit_guesses

    This method creates initial guesses for the fit parameters.
    See :ref:`curve_analysis_init_guess` for details.

    .. rubric:: _run_curve_fit

    This method performs the fitting with predefined fit models and the formatted dataset.
    This method internally calls the :meth:`_generate_fit_guesses` method.
    Note that this is a core functionality of the :meth:`_run_analysis` method,
    that creates fit result objects from the formatted dataset.

    .. rubric:: _evaluate_quality

    This method evaluates the quality of the fit based on the fit result.
    This returns "good" when reduced chi-squared is less than 3.0.
    Usually it returns string "good" or "bad" according to the evaluation.

    .. rubric:: _create_analysis_results

    This method creates analysis results for important fit parameters
    that might be defined by analysis options ``result_parameters``.

    .. rubric:: _create_curve_data

    This method creates analysis results containing the formatted dataset,
    i.e. data used for the fitting.
    Entries are created when the analysis option ``return_data_points`` is ``True``.
    If analysis consists of multiple series, an analysis result is created for
    each series definition.

    .. rubric:: _initialize

    This method initializes analysis options against input experiment data.
    Usually this method is called before other methods are called.

    """

    def __init__(
        self,
        models: Optional[List[lmfit.Model]] = None,
        name: Optional[str] = None,
    ):
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
            name: Optional. Name of this analysis.
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
            series_params = {}
            for series_def in self.__series__:
                models.append(
                    lmfit.Model(
                        name=series_def.name,
                        func=series_def.fit_func,
                        data_sort_key=series_def.filter_kwargs,
                    )
                )
                series_params[series_def.name] = {
                    "color": series_def.plot_color,
                    "symbol": series_def.plot_symbol,
                    "canvas": series_def.canvas,
                    "label": series_def.name,
                }
            self.plotter.set_figure_options(series_params=series_params)

        self._models = models or []
        self._name = name or self.__class__.__name__

        #: List[CurveData]: Processed experiment data set. For backward compatibility.
        self.__processed_data_set = {}

    @property
    def name(self) -> str:
        """Return name of this analysis."""
        return self._name

    @property
    def parameters(self) -> List[str]:
        """Return parameters of this curve analysis."""
        unite_params = []
        for model in self._models:
            for name in model.param_names:
                if name not in unite_params and name not in self.options.fixed_parameters:
                    unite_params.append(name)
        return unite_params

    @property
    def models(self) -> List[lmfit.Model]:
        """Return fit models."""
        return self._models

    def _run_data_processing(
        self,
        raw_data: List[Dict],
        models: List[lmfit.Model],
    ) -> pd.DataFrame:
        """Perform data processing from the experiment result payload.

        Args:
            raw_data: Payload in the experiment data.
            models: A list of LMFIT models that provide the model name and
                optionally data sorting keys.

        Returns:
            Curve analysis dataset.

        Raises:
            DataProcessorError: When model is a multi-objective function but
                data sorting option is not provided.
            DataProcessorError: When key for x values is not found in the metadata.
        """

        def _matched(metadata, **filters):
            try:
                return all(metadata[key] == val for key, val in filters.items())
            except KeyError:
                return False

        if not self.options.filter_data:
            analyzed_data = raw_data
        else:
            analyzed_data = [
                d for d in raw_data if _matched(d["metadata"], **self.options.filter_data)
            ]

        x_key = self.options.x_key

        try:
            xdata = np.asarray([datum["metadata"][x_key] for datum in analyzed_data], dtype=float)
        except KeyError as ex:
            raise DataProcessorError(
                f"X value key {x_key} is not defined in circuit metadata."
            ) from ex

        ydata = self.options.data_processor(analyzed_data)
        shots = np.asarray([datum.get("shots", np.nan) for datum in analyzed_data])

        if len(models) == 1:
            # all data belongs to the single model
            model_name = np.full(xdata.size, models[0]._name, dtype=object)
            model_index = np.full(xdata.size, 0, dtype=int)
        else:
            model_name = np.full(xdata.size, "unassigned", dtype=object)
            model_index = np.full(xdata.size, np.nan, dtype=int)

            for idx, sub_model in enumerate(models):
                try:
                    tags = sub_model.opts["data_sort_key"]
                except KeyError as ex:
                    raise DataProcessorError(
                        f"Data sort options for model {sub_model.name} is not defined."
                    ) from ex
                if tags is None:
                    continue
                matched_inds = np.asarray(
                    [_matched(d["metadata"], **tags) for d in analyzed_data], dtype=bool
                )
                model_name[matched_inds] = sub_model._name
                model_index[matched_inds] = idx

        to_df = {
            "x_val": xdata,
            "y_val": unp.nominal_values(ydata),
            "y_err": unp.std_devs(ydata),
            "samples": shots,
            "model_name": model_name,
            "model_index": model_index,
            "analysis_group": np.full(xdata.size, self._name, dtype=object),
            "data_kind": np.full(xdata.size, "raw", dtype=object)
        }

        return pd.DataFrame.from_dict(to_df)

    def _format_data(
        self,
        curve_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Postprocessing for the processed dataset.

        Args:
            curve_data: Curve analysis dataset.

        Returns:
            Curve analysis dataset with formatted entries.
        """
        # Average data
        grouped_by_model = curve_data.groupby(["model_index", "x_val"], as_index=False)
        if len(grouped_by_model) == len(curve_data):
            sorted_data = curve_data.sort_values("x_val", inplace=False)
            sorted_data.loc[:, "data_kind"] = "formatted"
            return pd.concat([curve_data, sorted_data])

        avg_methods = {
            "shots_weighted": shot_weighted_average,
            "sample": sample_average,
            "iwv": iwv_average,
        }
        # Data is automatically sorted by groupby operation.
        avg_data = grouped_by_model.apply(avg_methods[self.options.average_method])
        return pd.concat([curve_data, avg_data])

    def _generate_fit_guesses(
        self,
        user_opt: FitOptions,
        curve_data: pd.DataFrame,  # pylint: disable=unused-argument
    ) -> Union[FitOptions, List[FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        return user_opt

    def _run_curve_fit(
        self,
        curve_data: pd.DataFrame,
        models: List[lmfit.Model],
    ) -> CurveFitResult:
        """Perform curve fitting on given data collection and fit models.

        Args:
            curve_data: Curve analysis dataset.
            models: A list of LMFIT models that are used to build a cost function
                for the LMFIT minimizer.

        Returns:
            The best fitting outcome with minimum reduced chi-squared value.
        """
        # Patch the properties for backward compatibility
        cond = curve_data.data_kind == "formatted"
        data_to_fit = curve_data[cond]
        data_to_fit.__class__ = CurveData

        unite_parameter_names = []
        for model in models:
            # Seems like this is not efficient looping, but using set operation sometimes
            # yields bad fit. Not sure if this is an edge case, but
            # `TestRamseyXY` unittest failed due to the significant chisq value
            # in which the least_square fitter terminates with `xtol` rather than `ftol`
            # condition, i.e. `ftol` condition indicates termination by cost function.
            # This code respects the ordering of parameters so that it matches with
            # the signature of fit function and it is backward compatible.
            # In principle this should not matter since LMFIT maps them with names
            # rather than index. Need more careful investigation.
            for name in model.param_names:
                if name not in unite_parameter_names:
                    unite_parameter_names.append(name)

        default_fit_opt = FitOptions(
            parameters=unite_parameter_names,
            default_p0=self.options.p0,
            default_bounds=self.options.bounds,
            **self.options.lmfit_options,
        )

        # Bind fixed parameters if not empty
        if self.options.fixed_parameters:
            fixed_parameters = {
                k: v for k, v in self.options.fixed_parameters.items() if k in unite_parameter_names
            }
            default_fit_opt.p0.set_if_empty(**fixed_parameters)
        else:
            fixed_parameters = {}

        try:
            fit_options = self._generate_fit_guesses(default_fit_opt, data_to_fit)
        except TypeError:
            warnings.warn(
                "Calling '_generate_fit_guesses' method without curve data has been "
                "deprecated and will be prohibited after 0.4. "
                "Update the method signature of your custom analysis class.",
                DeprecationWarning,
            )
            # pylint: disable=no-value-for-parameter
            fit_options = self._generate_fit_guesses(default_fit_opt)
        if isinstance(fit_options, FitOptions):
            fit_options = [fit_options]

        valid_uncertainty = np.all(np.isfinite(data_to_fit.y_err))

        # Pre generate formatted data structure for speedup.
        sub_data = []
        for model in models:
            cond = data_to_fit.model_name == model._name
            model_data = data_to_fit[cond]
            sub_data.append(
                (
                    model_data.x_val.to_numpy(),
                    model_data.y_val.to_numpy(),
                    1.0 / model_data.y_err.to_numpy() if valid_uncertainty else None,
                )
            )

        # Objective function for minimize. This computes composite residuals of sub models.
        def _objective(_params):
            ys = []
            for model, (x, y, w) in zip(models, sub_data):
                yi = model._residual(params=_params, data=y, weights=w, x=x)
                ys.append(yi)
            return np.concatenate(ys)
        # Run fit for each configuration
        res = None
        for fit_option in fit_options:
            # Setup parameter configuration, i.e. init value, bounds
            guess_params = lmfit.Parameters()
            for name in unite_parameter_names:
                bounds = fit_option.bounds[name] or (-np.inf, np.inf)
                guess_params.add(
                    name=name,
                    value=fit_option.p0[name],
                    min=bounds[0],
                    max=bounds[1],
                    vary=name not in fixed_parameters,
                )

            try:
                new = lmfit.minimize(
                    fcn=_objective,
                    params=guess_params,
                    method=self.options.fit_method,
                    scale_covar=not valid_uncertainty,
                    nan_policy="omit",
                    **fit_option.fitter_opts,
                )
            except Exception:  # pylint: disable=broad-except
                continue

            if res is None or not res.success:
                res = new
                continue

            if new.success and res.redchi > new.redchi:
                res = new

        return convert_lmfit_result(res, models, data_to_fit.x_val, data_to_fit.y_val)

    def _draw_figures(
        self,
        curve_data: pd.DataFrame,
        fit_data: CurveFitResult,
        analysis_results: List[AnalysisResultData],
    ) -> List["pyplot.Figure"]:
        """Draw figures with experiment data and analysis data.

        Args:
            curve_data: Curve analysis dataset.
            fit_data: Result of fitting.
            analysis_results: Analyzed experimental quantities.

        Returns:
            Figures.
        """
        for model in self._models:
            model_name = model._name
            cond = curve_data.model_name == model_name
            sub_data = curve_data[cond]
            if len(sub_data) == 0:
                # If data is empty, skip drawing this model.
                # This is the case when fit model exist but no data to fit is provided.
                # For example, experiment may omit experimenting with some setting.
                continue
            if self.options.plot_raw_data:
                cond = sub_data.data_kind == "raw"
                raw_data = sub_data[cond]
                self.plotter.set_series_data(
                    model_name,
                    x=raw_data.x_val,
                    y=raw_data.y_val,
                )
            cond = sub_data.data_kind == "formatted"
            formatted_data = sub_data[cond]
            self.plotter.set_series_data(
                model_name,
                x_formatted=formatted_data.x_val,
                y_formatted=formatted_data.y_val,
                y_formatted_err=formatted_data.y_err,
            )
            if fit_data.success:
                # Draw fit line
                x_interp = np.linspace(
                    np.min(formatted_data.x_val), np.max(formatted_data.x_val), num=100
                )
                y_data_with_uncertainty = eval_with_uncertainties(
                    x=x_interp,
                    model=model,
                    params=fit_data.ufloat_params,
                )
                y_interp = unp.nominal_values(y_data_with_uncertainty)
                # Add fit line data
                self.plotter.set_series_data(
                    model_name,
                    x_interp=x_interp,
                    y_interp=y_interp,
                )
                if fit_data.covar is not None:
                    # Add confidence interval data
                    y_interp_err = unp.std_devs(y_data_with_uncertainty)
                    if np.isfinite(y_interp_err).all():
                        self.plotter.set_series_data(
                            model_name,
                            y_interp_err=y_interp_err,
                        )

        # Add supplementary data
        if fit_data.success:
            self.plotter.set_supplementary_data(fit_red_chi=fit_data.reduced_chisq)
            self.plotter.set_supplementary_data(primary_results=analysis_results)

        return [self.plotter.figure()]

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["pyplot.Figure"]]:

        # Prepare for fitting
        self._initialize(experiment_data)
        analysis_results = []

        # Run data processing
        curve_data = self._run_data_processing(
            raw_data=experiment_data.data(),
            models=self._models,
        )
        # Format data
        curve_data = self._format_data(curve_data)

        # Run fitting
        fit_data = self._run_curve_fit(
            curve_data=curve_data,
            models=self._models,
        )

        if fit_data.success:
            primary_results = self._create_analysis_results(
                fit_data=fit_data,
                quality=self._evaluate_quality(fit_data),
                **self.options.extra.copy(),
            )
            analysis_results.extend(primary_results)
        else:
            primary_results = []

        if self.options.return_fit_parameters:
            warnings.warn(
                "Now overview data is stored in the .artifacts of the ExperimentData. "
                "Enabling this option no longer create new analysis results. ",
                DeprecationWarning,
            )
        if self.options.return_data_points:
            warnings.warn(
                "Now curve data is stored in the .artifacts of the ExperimentData. "
                "Enabling this option no longer create new analysis results.",
                DeprecationWarning,
            )

        # Save dataset in artifacts
        # TODO composite analysis may override this entry.
        #  Need to define some structure.
        experiment_data.artifacts["curve_fit_overview"] = fit_data
        experiment_data.artifacts["curve_data"] = curve_data

        # Draw figures
        if self.options.plot:
            figures = self._draw_figures(curve_data, fit_data, primary_results)
        else:
            figures = []

        return analysis_results, figures

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
            tmp_mod = lmfit.Model(func=None)
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
