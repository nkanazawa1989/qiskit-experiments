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
A collection of functions that draw formatted curve analysis results.

For example, this visualization contains not only fit curves and raw data points,
but also some extra fitting information, such as fit values of some interesting parameters
and goodness of the fitting represented by chi-squared. These extra information can be
also visualized as a fit report.

Note that plotter is a class that only has a class method to draw the image.
This is just like a function, but allows serialization via Enum.
"""

from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
from matplotlib.ticker import FuncFormatter
from qiskit.utils import detach_prefix

from qiskit_experiments.curve_analysis.curve_data import SeriesDef, FitData, CurveData
from qiskit_experiments.framework import AnalysisResultData, UFloat
from qiskit_experiments.framework.matplotlib import get_non_gui_ax
from .curves import plot_scatter, plot_errorbar, plot_curve_fit
from .style import PlotterStyle


class MplDrawSingleCanvas:
    """A plotter to draw a single canvas figure for fit result."""

    @classmethod
    def draw(
        cls,
        series_defs: List[SeriesDef],
        raw_samples: List[CurveData],
        fit_samples: List[CurveData],
        tick_labels: Dict[str, str],
        fit_data: FitData,
        result_entries: List[AnalysisResultData],
        style: Optional[PlotterStyle] = None,
        axis: Optional["matplotlib.axes.Axes"] = None,
    ) -> "pyplot.Figure":
        """Create a fit result of all curves in the single canvas.

        Args:
            series_defs: List of definition for each curve.
            raw_samples: List of raw sample data for each curve.
            fit_samples: List of formatted sample data for each curve.
            tick_labels: Dictionary of axis label information. Axis units and label for x and y
                value should be explained.
            fit_data: fit data generated by the analysis.
            result_entries: List of analysis result data entries.
            style: Optional. A configuration object to modify the appearance of the figure.
            axis: Optional. A matplotlib Axis object.

        Returns:
            A matplotlib figure of the curve fit result.
        """
        if axis is None:
            axis = get_non_gui_ax()

            # update image size to experiment default
            figure = axis.get_figure()
            figure.set_size_inches(*style.figsize)
        else:
            figure = axis.get_figure()

        # draw all curves on the same canvas
        for series_def, raw_samp, fit_samp in zip(series_defs, raw_samples, fit_samples):
            draw_single_curve_mpl(
                axis=axis,
                series_def=series_def,
                raw_sample=raw_samp,
                fit_sample=fit_samp,
                fit_data=fit_data,
            )

        # add legend
        if len(series_defs) > 1:
            axis.legend(loc=style.legend_loc)

        # get axis scaling factor
        for this_axis in ("x", "y"):
            sub_axis = getattr(axis, this_axis + "axis")
            unit = tick_labels[this_axis + "val_unit"]
            label = tick_labels[this_axis + "label"]
            if unit:
                maxv = np.max(np.abs(sub_axis.get_data_interval()))
                scaled_maxv, prefix = detach_prefix(maxv, decimal=3)
                prefactor = scaled_maxv / maxv
                # pylint: disable=cell-var-from-loop
                sub_axis.set_major_formatter(FuncFormatter(lambda x, p: f"{x * prefactor: .3g}"))
                sub_axis.set_label_text(f"{label} [{prefix}{unit}]", fontsize=style.axis_label_size)
            else:
                sub_axis.set_label_text(label, fontsize=style.axis_label_size)
                axis.ticklabel_format(axis=this_axis, style="sci", scilimits=(-3, 3))

        if tick_labels["xlim"]:
            axis.set_xlim(tick_labels["xlim"])

        if tick_labels["ylim"]:
            axis.set_ylim(tick_labels["ylim"])

        # write analysis report
        if fit_data:
            report_str = write_fit_report(result_entries)
            report_str += r"Fit $\chi^2$ = " + f"{fit_data.reduced_chisq: .4g}"

            report_handler = axis.text(
                *style.fit_report_rpos,
                report_str,
                ha="center",
                va="top",
                size=style.fit_report_text_size,
                transform=axis.transAxes,
            )

            bbox_props = dict(boxstyle="square, pad=0.3", fc="white", ec="black", lw=1, alpha=0.8)
            report_handler.set_bbox(bbox_props)

        axis.tick_params(labelsize=style.tick_label_size)
        axis.grid(True)

        return figure


class MplDrawMultiCanvasVstack:
    """A plotter to draw a vertically stacked multi canvas figure for fit result."""

    @classmethod
    def draw(
        cls,
        series_defs: List[SeriesDef],
        raw_samples: List[CurveData],
        fit_samples: List[CurveData],
        tick_labels: Dict[str, str],
        fit_data: FitData,
        result_entries: List[AnalysisResultData],
        style: Optional[PlotterStyle] = None,
        axis: Optional["matplotlib.axes.Axes"] = None,
    ) -> "pyplot.Figure":
        """Create a fit result of all curves in the single canvas.

        Args:
            series_defs: List of definition for each curve.
            raw_samples: List of raw sample data for each curve.
            fit_samples: List of formatted sample data for each curve.
            tick_labels: Dictionary of axis label information. Axis units and label for x and y
                value should be explained.
            fit_data: fit data generated by the analysis.
            result_entries: List of analysis result data entries.
            style: Optional. A configuration object to modify the appearance of the figure.
            axis: Optional. A matplotlib Axis object.

        Returns:
            A matplotlib figure of the curve fit result.
        """
        if axis is None:
            axis = get_non_gui_ax()

            # update image size to experiment default
            figure = axis.get_figure()
            figure.set_size_inches(*style.figsize)
        else:
            figure = axis.get_figure()

        # get canvas number
        n_subplots = max(series_def.canvas for series_def in series_defs) + 1

        # use inset axis. this allows us to draw multiple canvases on a given single axis object
        inset_ax_h = (1 - (0.05 * (n_subplots - 1))) / n_subplots
        inset_axes = [
            axis.inset_axes(
                [0, 1 - (inset_ax_h + 0.05) * n_axis - inset_ax_h, 1, inset_ax_h],
                transform=axis.transAxes,
                zorder=1,
            )
            for n_axis in range(n_subplots)
        ]

        # show x label only in the bottom canvas
        for inset_axis in inset_axes[:-1]:
            inset_axis.set_xticklabels([])
        inset_axes[-1].get_shared_x_axes().join(*inset_axes)

        # remove original axis frames
        axis.spines.right.set_visible(False)
        axis.spines.left.set_visible(False)
        axis.spines.top.set_visible(False)
        axis.spines.bottom.set_visible(False)
        axis.set_xticks([])
        axis.set_yticks([])

        # collect data source per canvas
        plot_map = defaultdict(list)
        for curve_ind, series_def in enumerate(series_defs):
            plot_map[series_def.canvas].append(curve_ind)

        y_labels = tick_labels["ylabel"].split(",")
        if len(y_labels) == 1:
            y_labels = y_labels * n_subplots

        for ax_ind, curve_inds in plot_map.items():
            inset_axis = inset_axes[ax_ind]

            for curve_ind in curve_inds:
                draw_single_curve_mpl(
                    axis=inset_axis,
                    series_def=series_defs[curve_ind],
                    raw_sample=raw_samples[curve_ind],
                    fit_sample=fit_samples[curve_ind],
                    fit_data=fit_data,
                )

            # add legend to each inset axis
            if len(curve_inds) > 1:
                inset_axis.legend(loc=style.legend_loc)

            # format y axis tick value of each inset axis
            yaxis = getattr(inset_axis, "yaxis")
            unit = tick_labels["yval_unit"]
            label = y_labels[ax_ind]
            if unit:
                maxv = np.max(np.abs(yaxis.get_data_interval()))
                scaled_maxv, prefix = detach_prefix(maxv, decimal=3)
                prefactor = scaled_maxv / maxv
                # pylint: disable=cell-var-from-loop
                yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x * prefactor: .3g}"))
                yaxis.set_label_text(f"{label} [{prefix}{unit}]", fontsize=style.axis_label_size)
            else:
                inset_axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
                yaxis.set_label_text(label, fontsize=style.axis_label_size)

            if tick_labels["ylim"]:
                inset_axis.set_ylim(tick_labels["ylim"])

        # format x axis
        xaxis = getattr(inset_axes[-1], "xaxis")
        unit = tick_labels["xval_unit"]
        label = tick_labels["xlabel"]
        if unit:
            maxv = np.max(np.abs(xaxis.get_data_interval()))
            scaled_maxv, prefix = detach_prefix(maxv, decimal=3)
            prefactor = scaled_maxv / maxv
            # pylint: disable=cell-var-from-loop
            xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x * prefactor: .3g}"))
            xaxis.set_label_text(f"{label} [{prefix}{unit}]", fontsize=style.axis_label_size)
        else:
            axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
            xaxis.set_label_text(label, fontsize=style.axis_label_size)

        if tick_labels["xlim"]:
            inset_axes[-1].set_xlim(tick_labels["xlim"])

        # write analysis report
        if fit_data:
            report_str = write_fit_report(result_entries)
            report_str += r"Fit $\chi^2$ = " + f"{fit_data.reduced_chisq: .4g}"

            report_handler = axis.text(
                *style.fit_report_rpos,
                report_str,
                ha="center",
                va="top",
                size=style.fit_report_text_size,
                transform=axis.transAxes,
            )

            bbox_props = dict(boxstyle="square, pad=0.3", fc="white", ec="black", lw=1, alpha=0.8)
            report_handler.set_bbox(bbox_props)

        axis.tick_params(labelsize=style.tick_label_size)
        axis.grid(True)

        return figure


def draw_single_curve_mpl(
    axis: "matplotlib.axes.Axes",
    series_def: SeriesDef,
    raw_sample: CurveData,
    fit_sample: CurveData,
    fit_data: FitData,
):
    """A function that draws a single curve on the given plotter canvas.

    Args:
        axis: Drawer canvas.
        series_def: Definition of the curve to draw.
        raw_sample: Raw sample data.
        fit_sample: Formatted sample data.
        fit_data: Fitting parameter collection.
    """

    # plot raw data if data is formatted
    if not np.array_equal(raw_sample.y, fit_sample.y):
        plot_scatter(xdata=raw_sample.x, ydata=raw_sample.y, ax=axis, zorder=0)

    # plot formatted data
    if np.all(np.isnan(fit_sample.y_err)):
        sigma = None
    else:
        sigma = np.nan_to_num(fit_sample.y_err)

    plot_errorbar(
        xdata=fit_sample.x,
        ydata=fit_sample.y,
        sigma=sigma,
        ax=axis,
        label=series_def.name,
        marker=series_def.plot_symbol,
        color=series_def.plot_color,
        zorder=1,
        linestyle="",
    )

    # plot fit curve
    if fit_data:
        plot_curve_fit(
            func=series_def.fit_func,
            result=fit_data,
            ax=axis,
            color=series_def.plot_color,
            zorder=2,
        )


def write_fit_report(result_entries: List[AnalysisResultData]) -> str:
    """A function that generates fit reports documentation from list of data.

    Args:
        result_entries: List of data entries.

    Returns:
        Documentation of fit reports.
    """
    analysis_description = ""

    def format_val(float_val: float) -> str:
        if np.abs(float_val) < 1e-3 or np.abs(float_val) > 1e3:
            return f"{float_val: .4e}"
        return f"{float_val: .4g}"

    for res in result_entries:
        if isinstance(res.value, UFloat):
            fitval = res.value
            if fitval.tag:
                # unit is defined. do detaching prefix, i.e. 1000 Hz -> 1 kHz
                val, val_prefix = detach_prefix(fitval.nominal_value, decimal=3)
                val_unit = val_prefix + fitval.tag
                value_repr = f"{val: .3g}"

                # write error bar if it is finite value
                if fitval.std_dev is not None and np.isfinite(fitval.std_dev):
                    # with stderr
                    err, err_prefix = detach_prefix(fitval.std_dev, decimal=3)
                    err_unit = err_prefix + fitval.tag
                    if val_unit == err_unit:
                        # same value scaling, same prefix
                        value_repr += f" \u00B1 {err: .2f} {val_unit}"
                    else:
                        # different value scaling, different prefix
                        value_repr += f" {val_unit} \u00B1 {err: .2f} {err_unit}"
                else:
                    # without stderr, just append unit
                    value_repr += f" {val_unit}"
            else:
                # unit is not defined. raw value formatting is performed.
                value_repr = format_val(fitval.nominal_value)
                if np.isfinite(fitval.std_dev):
                    # with stderr
                    value_repr += f" \u00B1 {format_val(fitval.std_dev)}"

            analysis_description += f"{res.name} = {value_repr}\n"

    return analysis_description
