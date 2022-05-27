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

"""
Base class of curve analysis.
"""

import warnings

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from uncertainties import unumpy as unp
from lmfit import Model, Parameters, minimize

from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, Options, ExperimentData
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.data_processing.processor_library import get_processor
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.warnings import deprecated_function

from .curve_data import CurveData, ParameterRepr, FitOptions, CurveFitResult
from .visualization import MplCurveDrawer, BaseCurveDrawer
from .utils import convert_lmfit_result
from .format_data import shot_weighted_average

PARAMS_ENTRY_PREFIX = "@Parameters_"
DATA_ENTRY_PREFIX = "@Data_"


class BaseCurveAnalysis(BaseAnalysis, ABC):
    """Abstract superclass of curve analysis base classes.

    Note that this class doesn't define :meth:`_run_analysis` method,
    and no actual fitting protocol is implemented in this base class.
    However, this class defines several common methods that can be reused.
    A curve analysis subclass can construct proper fitting protocol
    by combining following methods, i.e. subroutines.
    See :ref:`curve_analysis_workflow` for how these subroutines are called.

    .. rubric:: _generate_fit_guesses

    This method creates initial guesses for the fit parameters.
    This might be overridden by subclass.
    See :ref:`curve_analysis_init_guess` for details.

    .. rubric:: _format_data

    This method consumes the processed dataset and outputs the formatted dataset.
    By default, this method takes the average of y values over
    the same x values and then sort the entire data by x values.

    .. rubric:: _evaluate_quality

    This method evaluates the quality of the fit based on the fit result.
    This returns "good" when reduced chi-squared is less than 3.0.
    Usually it returns string "good" or "bad" according to the evaluation.
    This criterion can be updated by subclass.

    .. rubric:: _run_data_processing

    This method performs data processing and returns the processed dataset.
    By default, it internally calls :class:`DataProcessor` instance from the analysis options
    and processes experiment data payload to create Y data with uncertainty.
    X data and other metadata are generated within this method by inspecting the
    circuit metadata. The series classification is also performed by based upon the
    matching of circuit metadata and :attr:`SeriesDef.filter_kwargs`.

    .. rubric:: _run_curve_fit

    This method performs the fitting with predefined fit models and the formatted dataset.
    This method internally calls :meth:`_generate_fit_guesses` method.
    Note that this is a core functionality of the :meth:`_run_analysis` method,
    that creates fit result object from the formatted dataset.

    .. rubric:: _create_analysis_results

    This method to creates analysis results for important fit parameters
    that might be defined by analysis options ``result_parameters``.
    In addition, another entry for all fit parameters is created when
    the analysis option ``return_fit_parameters`` is ``True``.

    .. rubric:: _create_curve_data

    This method to creates analysis results for the formatted dataset, i.e. data used for the fitting.
    Entries are created when the analysis option ``return_data_points`` is ``True``.
    If analysis consists of multiple series, analysis result is created for
    each curve data in the series definitions.

    .. rubric:: _initialize

    This method initializes analysis options against input experiment data.
    Usually this method is called before other methods are called.

    """

    @property
    @abstractmethod
    def parameters(self) -> List[str]:
        """Return parameters estimated by this analysis."""

    @property
    def drawer(self) -> BaseCurveDrawer:
        """A short-cut for curve drawer instance."""
        return self._options.curve_drawer

    @classmethod
    def _default_options(cls) -> Options:
        """Return default analysis options.

        Analysis Options:
            curve_drawer (BaseCurveDrawer): A curve drawer instance to visualize
                the analysis result.
            plot_raw_data (bool): Set ``True`` to draw processed data points,
                dataset without formatting, on canvas. This is ``False`` by default.
            plot (bool): Set ``True`` to create figure for fit result.
                This is ``True`` by default.
            return_fit_parameters (bool): Set ``True`` to return all fit model parameters
                with details of the fit outcome. Default to ``True``.
            data_processor (Callable): A callback function to format experiment data.
                This can be a :class:`~qiskit_experiments.data_processing.DataProcessor`
                instance that defines the `self.__call__` method.
            normalization (bool) : Set ``True`` to normalize y values within range [-1, 1].
                Default to ``False``.
            p0 (Dict[str, float]): Initial guesses for the fit parameters.
                The dictionary is keyed on the fit parameter names.
            bounds (Dict[str, Tuple[float, float]]): Boundary of fit parameters.
                The dictionary is keyed on the fit parameter names and
                values are the tuples of (min, max) of each parameter.
            fit_method (str): Fit method that LMFIT minimizer uses.
                Default to ``least_squares`` method which implements the
                Trust Region Reflective algorithm to solve the minimization problem.
                See LMFIT documentation for available options.
            curve_fitter_options (Dict[str, Any]) Options that are passed to the
                LMFIT minimizer. Acceptable options depend on fit_method.
            x_key (str): Circuit metadata key representing a scanned value.
            result_parameters (List[Union[str, ParameterRepr]): Parameters reported in the
                database as a dedicated entry. This is a list of parameter representation
                which is either string or ParameterRepr object. If you provide more
                information other than name, you can specify
                ``[ParameterRepr("alpha", "\u03B1", "a.u.")]`` for example.
                The parameter name should be defined in the series definition.
                Representation should be printable in standard output, i.e. no latex syntax.
            extra (Dict[str, Any]): A dictionary that is appended to all database entries
                as extra information.
            fixed_parameters (Dict[str, Any]): Fitting model parameters that are fixed
                during the curve fitting. This should be provided with default value
                keyed on one of the parameter names in the series definition.
        """
        options = super()._default_options()

        options.curve_drawer = MplCurveDrawer()
        options.plot_raw_data = False
        options.plot = True
        options.return_fit_parameters = True
        options.data_processor = None
        options.normalization = False
        options.x_key = "xval"
        options.result_parameters = []
        options.extra = {}
        options.fit_method = "least_squares"
        options.curve_fitter_options = {}
        options.p0 = {}
        options.bounds = {}
        options.fixed_parameters = {}

        # Set automatic validator for particular option values
        options.set_validator(field="data_processor", validator_value=DataProcessor)
        options.set_validator(field="curve_drawer", validator_value=BaseCurveDrawer)

        return options

    def set_options(self, **fields):
        """Set the analysis options for :meth:`run` method.

        Args:
            fields: The fields to update the options

        Raises:
            KeyError: When removed option ``curve_fitter`` is set.
        """
        # TODO remove this in Qiskit Experiments v0.4
        if "curve_plotter" in fields:
            warnings.warn(
                "The analysis option 'curve_plotter' has been deprecated. "
                "The option is replaced with 'curve_drawer' that takes 'MplCurveDrawer' instance. "
                "If this is a loaded analysis, please save this instance again to update option value. "
                "The 'curve_plotter' argument along with this warning will be removed "
                "in Qiskit Experiments 0.4.",
                DeprecationWarning,
                stacklevel=2,
            )
            del fields["curve_plotter"]

        if "curve_fitter" in fields:
            warnings.warn(
                "Setting curve fitter to analysis options has been deprecated and "
                "the option has been removed. The fitter setting is dropped. "
                "Now you can directly override '_run_curve_fit' method to apply custom fitter. "
                "The `curve_fitter` argument along with this warning will be removed "
                "in Qiskit Experiments 0.4.",
                DeprecationWarning,
                stacklevel=2,
            )
            del fields["curve_fitter"]

        if "return_data_points" in fields:
            warnings.warn(
                "Option 'return_data_points' is removed. Now raw experiment data is "
                "stored as a part of @Parameters entry which is a 'CurveFitResult' dataclass."
                "CurveFitResult.data provides a full observed data in dataframe format."
            )
            del fields["return_data_points"]

        # pylint: disable=no-member
        draw_options = set(self.drawer.options.__dict__.keys()) | {"style"}
        deprecated = draw_options & fields.keys()
        if any(deprecated):
            warnings.warn(
                f"Option(s) {deprecated} have been moved to draw_options and will be removed soon. "
                "Use self.drawer.set_options instead. "
                "If this is a loaded analysis, please save this instance again to update option value. "
                "These arguments along with this warning will be removed "
                "in Qiskit Experiments 0.4.",
                DeprecationWarning,
                stacklevel=2,
            )
            draw_options = dict()
            for depopt in deprecated:
                if depopt == "style":
                    for k, v in fields.pop("style").items():
                        draw_options[k] = v
                else:
                    draw_options[depopt] = fields.pop(depopt)
            self.drawer.set_options(**draw_options)

        super().set_options(**fields)

    def _generate_fit_guesses(
        self,
        user_opt: FitOptions,
        curve_data: pd.DataFrame,  # pylint: disable=unused-argument
    ) -> Union[FitOptions, List[FitOptions]]:
        """Create algorithmic guess with analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        return user_opt

    def _format_data(
        self,
        curve_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Postprocessing for the processed dataset.

        Args:
            curve_data: Processed dataset created from experiment results.

        Returns:
            Formatted data frame.
        """
        # This returns DataFrameGroupBy object.
        # Sort dataframe with model and x.
        # Take average over the same x values in the same group.
        grouped_by_model = curve_data.groupby(["model", "x"], as_index=False)
        averaged_df = grouped_by_model.apply(shot_weighted_average)

        return averaged_df

    def _evaluate_quality(
        self,
        fit_data: CurveFitResult,
    ) -> Union[str, None]:
        """Evaluate quality of the fit result.

        Args:
            fit_data: Fit outcome.

        Returns:
            String that represents fit result quality. Usually "good" or "bad".
        """
        if fit_data.reduced_chisq < 3.0:
            return "good"
        return "bad"

    def _run_data_processing(
        self,
        raw_data: List[Dict],
        models: List[Model],
    ) -> pd.DataFrame:
        """Perform data processing from the experiment result payload.

        Args:
            raw_data: Payload in the experiment data.
            models: LMFIT models to provide data sort conditions.

        Returns:
            Processed data frame that will be sent to the formatter method.
        """
        num_data = len(raw_data)

        xdata = np.full(num_data, fill_value=np.nan)
        shots = np.zeros(num_data)
        extra = defaultdict(lambda: [pd.NA] * num_data)
        ydata = self.options.data_processor(raw_data)

        for i, datum in enumerate(raw_data):
            meta = datum["metadata"]
            for k, v in meta.items():
                if k == self.options.x_key:
                    xdata[i] = v
                else:
                    extra[k][i] = v
            shots[i] = datum.get("shots", np.nan)

        results = pd.DataFrame.from_dict(
            {
                "x": xdata,
                "y": unp.nominal_values(ydata),
                "y_err": unp.std_devs(ydata),
                "shots": shots,
                "model": [pd.NA] * num_data,
                **extra
            }
        )

        if len(models) > 1:
            # Tag data series with model name
            for model in models:
                try:
                    conds = [f"{k} == {repr(v)}" for k, v in model.opts["data_sort_key"].items()]
                except KeyError as ex:
                    raise DataProcessorError(
                        f"Data sort key for model '{model._name}' is not defined."
                    ) from ex
                # Colum index 4 is "model" thus updating the model column here
                results.iloc[results.query(" & ".join(conds)).index, 4] = model._name
        else:
            results["model"] = models[0]._name

        # Lastly data sort by x column values.
        # This is important because fit guess function assumes monotonically increase x vals.
        # When we combine multiple experiment data for analysis, x vals are not necessary
        # arranged in ascending order.
        results.sort_values(by="x", inplace=True)

        return results

    def _run_curve_fit(
        self,
        curve_data: pd.DataFrame,
        models: List[Model],
    ) -> CurveFitResult:
        """Perform curve fitting on given data collection and fit models.

        Args:
            curve_data: Formatted data frame to fit.
            models: A list of LMFIT models that are used to build a cost function
                for the LMFIT minimizer.

        Returns:
            The best fitting outcome with minimum reduced chi-squared value.
        """
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
            **self.options.curve_fitter_options,
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
            fit_options = self._generate_fit_guesses(default_fit_opt, curve_data)
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

        # Prepare default fitting arguments (can be overridden by fit options)
        valid_uncertainty = np.all(np.isfinite(curve_data.y_err))

        # Objective function for minimize. This computes composite residuals of sub models.
        grouped_data = curve_data.groupby("model")

        def _objective(_params):
            ys = []
            for model in models:
                sub_data = grouped_data.get_group(model._name)
                yi = model._residual(
                    params=_params,
                    data=sub_data.y,
                    weights=1.0 / sub_data.y_err if valid_uncertainty else None,
                    x=sub_data.x,
                )
                ys.append(yi)
            return np.concatenate(ys)

        # Run fit for each configuration
        res = None
        for fit_option in fit_options:
            # Setup parameter configuration, i.e. init value, bounds
            guess_params = Parameters()
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
                new = minimize(
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

        return convert_lmfit_result(res, models, curve_data)

    def _create_analysis_results(
        self,
        fit_data: CurveFitResult,
        quality: str,
        **metadata,
    ) -> List[AnalysisResultData]:
        """Create analysis results for important fit parameters.

        Args:
            fit_data: Fit outcome.
            quality: Quality of fit outcome.

        Returns:
            List of analysis result data.
        """
        outcomes = []

        # Create entries for important parameters
        for param_repr in self.options.result_parameters:
            if isinstance(param_repr, ParameterRepr):
                p_name = param_repr.name
                p_repr = param_repr.repr or param_repr.name
                unit = param_repr.unit
            else:
                p_name = param_repr
                p_repr = param_repr
                unit = None

            if unit:
                par_metadata = metadata.copy()
                par_metadata["unit"] = unit
            else:
                par_metadata = metadata

            outcome = AnalysisResultData(
                name=p_repr,
                value=fit_data.ufloat_params[p_name],
                chisq=fit_data.reduced_chisq,
                quality=quality,
                extra=par_metadata,
            )
            outcomes.append(outcome)

        return outcomes

    @deprecated_function(
        "0.5",
        msg="No need to create separate entry for curve data. This is a part of @Parameters entry."
    )
    def _create_curve_data(
        self,
        curve_data: CurveData,
        models: List[Model],
        **metadata,
    ) -> List[AnalysisResultData]:
        """Create analysis results for raw curve data.

        Args:
            curve_data: Formatted data that is used for the fitting.
            models: A list of LMFIT models that provides model names
                to extract subsets of experiment data.

        Returns:
            List of analysis result data.
        """

        return []

    def _initialize(
        self,
        experiment_data: ExperimentData,
    ):
        """Initialize curve analysis with experiment data.

        This method is called ahead of other processing.

        Args:
            experiment_data: Experiment data to analyze.
        """
        # Initialize canvas
        if self.options.plot:
            self.drawer.initialize_canvas()

        # Initialize data processor
        # TODO move this to base analysis in follow-up
        data_processor = self.options.data_processor or get_processor(experiment_data, self.options)

        if not data_processor.is_trained:
            data_processor.train(data=experiment_data.data())
        self.set_options(data_processor=data_processor)
