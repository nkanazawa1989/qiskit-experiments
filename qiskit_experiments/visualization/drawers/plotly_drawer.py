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

"""Curve drawer for plotly backend."""

from typing import Any, Dict, Optional, Sequence, Tuple

from .base_drawer import BaseDrawer

import numpy as np

import plotly.graph_objects as go


class PlotlyDrawer(BaseDrawer):
    """Drawer for Plotly backend."""

    # DefaultMarkers = MarkerStyle.filled_markers
    # DefaultColors = tab10.colors
    #
    # class PrefixFormatter(Formatter):
    #     """Matplotlib axis formatter to detach prefix.
    #
    #     If a value is, e.g., x=1000.0 and the factor is 1000, then it will be shown
    #     as 1.0 in the ticks and its unit will be shown with the prefactor 'k'
    #     in the axis label.
    #     """
    #
    #     def __init__(self, factor: float):
    #         """Create a PrefixFormatter instance.
    #
    #         Args:
    #             factor: factor by which to scale tick values.
    #         """
    #         self.factor = factor
    #
    #     def __call__(self, x: Any, pos: int = None) -> str:
    #         """Returns the formatted string for tick position ``pos`` and value ``x``.
    #
    #         Args:
    #             x: the tick value to format.
    #             pos: the tick label position.
    #
    #         Returns:
    #             str: the formatted tick label.
    #         """
    #         return self.fix_minus("{:.3g}".format(x * self.factor))

    def __init__(self):
        super().__init__()
        # Used to track which series have already been plotted. Needed for _get_default_marker and
        # _get_default_color.
        self._series = list()

    @classmethod
    def _default_options(cls):
        options = super()._default_options()
        options.update_options(
            dpi=70,
            template="plotly_white",
        )
        return options

    def _translate_legend_vloc(self, loc: str):
        if loc == "upper":
            return {"yanchor": "top", "y": 0.99}
        if loc == "lower":
            return {"yanchor": "bottom", "y": 0.01}
        if loc == "center":
            return {"yanchor": "middle", "y": 0.5}
        raise ValueError("Not valid legend location.")

    def _translate_legend_hloc(self, loc: str):
        if loc == "left":
            return {"xanchor": "left", "x": 0.01}
        if loc == "right":
            return {"xanchor": "right", "x": 0.99}
        if loc == "center":
            return {"xanchor": "center", "x": 0.5}
        raise ValueError("Not valid legend location.")

    def initialize_canvas(self):
        # TODO support user provided axis and multi-canvas
        w_inch, h_inch = self.style["figsize"]
        wsize = w_inch * self.options.dpi
        hsize = h_inch * self.options.dpi
        v_legend_loc, h_legend_loc = self.style["legend_loc"].split(" ")

        layout = go.Layout(
            width=wsize,
            height=hsize,
            xaxis=go.layout.XAxis(
                linewidth=1,
                zeroline=False,
                mirror=True,
            ),
            yaxis=go.layout.YAxis(
                linewidth=1,
                zeroline=False,
                mirror=True,
            ),
            paper_bgcolor="rgba(0, 0, 0, 0)",
            legend=go.layout.Legend(
                **self._translate_legend_vloc(v_legend_loc),
                **self._translate_legend_hloc(h_legend_loc),
            ),
            margin=go.layout.Margin(
                l=wsize * 0.05,
                r=wsize * 0.05,
                t=hsize * 0.05,
                b=hsize * 0.05,
            ),
            hovermode='x',
            modebar=go.layout.Modebar(
                add=["drawline", "drawopenpath", "eraseshape"],
            ),
            template=self.options.template,
        )
        self._axis = go.Figure(layout=layout)

        #
        # # Create axis if empty
        # if not self.options.axis:
        #     axis = get_non_gui_ax()
        #     figure = axis.get_figure()
        #     figure.set_size_inches(*self.style["figsize"])
        # else:
        #     axis = self.options.axis
        #
        # n_rows, n_cols = self.options.subplots
        # n_subplots = n_cols * n_rows
        # if n_subplots > 1:
        #     # Add inset axis. User may provide a single axis object via the analysis option,
        #     # while this analysis tries to draw its result in multiple canvases,
        #     # especially when the analysis consists of multiple curves.
        #     # Inset axis is experimental implementation of matplotlib 3.0 so maybe unstable API.
        #     # This draws inset axes with shared x and y axis.
        #     inset_ax_h = 1 / n_rows
        #     inset_ax_w = 1 / n_cols
        #     for i in range(n_rows):
        #         for j in range(n_cols):
        #             # x0, y0, width, height
        #             bounds = [
        #                 inset_ax_w * j,
        #                 1 - inset_ax_h * (i + 1),
        #                 inset_ax_w,
        #                 inset_ax_h,
        #             ]
        #             sub_ax = axis.inset_axes(bounds, transform=axis.transAxes, zorder=1)
        #             if j != 0:
        #                 # remove y axis except for most-left plot
        #                 sub_ax.set_yticklabels([])
        #             else:
        #                 # this axis locates at left, write y-label
        #                 if self.figure_options.ylabel:
        #                     label = self.figure_options.ylabel
        #                     if isinstance(label, list):
        #                         # Y label can be given as a list for each sub axis
        #                         label = label[i]
        #                     sub_ax.set_ylabel(label, fontsize=self.style["axis_label_size"])
        #             if i != n_rows - 1:
        #                 # remove x axis except for most-bottom plot
        #                 sub_ax.set_xticklabels([])
        #             else:
        #                 # this axis locates at bottom, write x-label
        #                 if self.figure_options.xlabel:
        #                     label = self.figure_options.xlabel
        #                     if isinstance(label, list):
        #                         # X label can be given as a list for each sub axis
        #                         label = label[j]
        #                     sub_ax.set_xlabel(label, fontsize=self.style["axis_label_size"])
        #             if j == 0 or i == n_rows - 1:
        #                 # Set label size for outer axes where labels are drawn
        #                 sub_ax.tick_params(labelsize=self.style["tick_label_size"])
        #             sub_ax.grid()
        #
        #     # Remove original axis frames
        #     axis.axis("off")
        # else:
        #     axis.set_xlabel(self.figure_options.xlabel, fontsize=self.style["axis_label_size"])
        #     axis.set_ylabel(self.figure_options.ylabel, fontsize=self.style["axis_label_size"])
        #     axis.tick_params(labelsize=self.style["tick_label_size"])
        #     axis.grid()
        #
        # self._axis = axis

    def format_canvas(self):
        self._axis.update_xaxes(
            title=self.figure_options.xlabel,
            title_font_size=self.style["axis_label_size"],
        )
        self._axis.update_yaxes(
            title=self.figure_options.ylabel,
            title_font_size=self.style["axis_label_size"],
        )

        # if self._axis.child_axes:
        #     # Multi canvas mode
        #     all_axes = self._axis.child_axes
        # else:
        #     all_axes = [self._axis]
        #
        # # Add data labels if there are multiple labels registered per sub_ax.
        # for sub_ax in all_axes:
        #     _, labels = sub_ax.get_legend_handles_labels()
        #     if len(labels) > 1:
        #         sub_ax.legend(loc=self.style["legend_loc"])
        #
        # # Format x and y axis
        # for ax_type in ("x", "y"):
        #     # Get axis formatter from drawing options
        #     if ax_type == "x":
        #         lim = self.figure_options.xlim
        #         unit = self.figure_options.xval_unit
        #     else:
        #         lim = self.figure_options.ylim
        #         unit = self.figure_options.yval_unit
        #
        #     # Compute data range from auto scale
        #     if not lim:
        #         v0 = np.nan
        #         v1 = np.nan
        #         for sub_ax in all_axes:
        #             if ax_type == "x":
        #                 this_v0, this_v1 = sub_ax.get_xlim()
        #             else:
        #                 this_v0, this_v1 = sub_ax.get_ylim()
        #             v0 = np.nanmin([v0, this_v0])
        #             v1 = np.nanmax([v1, this_v1])
        #         lim = (v0, v1)
        #
        #     # Format axis number notation
        #     if unit:
        #         # If value is specified, automatically scale axis magnitude
        #         # and write prefix to axis label, i.e. 1e3 Hz -> 1 kHz
        #         maxv = max(np.abs(lim[0]), np.abs(lim[1]))
        #         try:
        #             scaled_maxv, prefix = detach_prefix(maxv, decimal=3)
        #             prefactor = scaled_maxv / maxv
        #         except ValueError:
        #             prefix = ""
        #             prefactor = 1
        #
        #         formatter = MplDrawer.PrefixFormatter(prefactor)
        #         units_str = f" [{prefix}{unit}]"
        #     else:
        #         # Use scientific notation with 3 digits, 1000 -> 1e3
        #         formatter = ScalarFormatter()
        #         formatter.set_scientific(True)
        #         formatter.set_powerlimits((-3, 3))
        #
        #         units_str = ""
        #
        #     for sub_ax in all_axes:
        #         if ax_type == "x":
        #             ax = getattr(sub_ax, "xaxis")
        #             tick_labels = sub_ax.get_xticklabels()
        #         else:
        #             ax = getattr(sub_ax, "yaxis")
        #             tick_labels = sub_ax.get_yticklabels()
        #
        #         if tick_labels:
        #             # Set formatter only when tick labels exist
        #             ax.set_major_formatter(formatter)
        #         if units_str:
        #             # Add units to label if both exist
        #             label_txt_obj = ax.get_label()
        #             label_str = label_txt_obj.get_text()
        #             if label_str:
        #                 label_txt_obj.set_text(label_str + units_str)
        #
        #     # Auto-scale all axes to the first sub axis
        #     if ax_type == "x":
        #         # get_shared_y_axes() is immutable from matplotlib>=3.6.0. Must use Axis.sharey()
        #         # instead, but this can only be called once per axis. Here we call sharey  on all axes in
        #         # a chain, which should have the same effect.
        #         if len(all_axes) > 1:
        #             for ax1, ax2 in zip(all_axes[1:], all_axes[0:-1]):
        #                 ax1.sharex(ax2)
        #         all_axes[0].set_xlim(lim)
        #     else:
        #         # get_shared_y_axes() is immutable from matplotlib>=3.6.0. Must use Axis.sharey()
        #         # instead, but this can only be called once per axis. Here we call sharey  on all axes in
        #         # a chain, which should have the same effect.
        #         if len(all_axes) > 1:
        #             for ax1, ax2 in zip(all_axes[1:], all_axes[0:-1]):
        #                 ax1.sharey(ax2)
        #         all_axes[0].set_ylim(lim)
        #
        # # Add title
        # if self.figure_options.figure_title is not None:
        #     self._axis.set_title(
        #         label=self.figure_options.figure_title,
        #         fontsize=self.style["axis_label_size"],
        #     )

    def _get_axis(self, index: Optional[int] = None):
        """A helper method to get inset axis.

        Args:
            index: Index of inset axis. If nothing is provided, it returns the entire axis.

        Returns:
            Corresponding axis object.

        Raises:
            IndexError: When axis index is specified but no inset axis is found.
        """
        # if index is not None:
        #     try:
        #         return self._axis.child_axes[index]
        #     except IndexError as ex:
        #         raise IndexError(
        #             f"Canvas index {index} is out of range. "
        #             f"Only {len(self._axis.child_axes)} subplots are initialized."
        #         ) from ex
        # else:
        return self._axis

    # def _get_default_color(self, name: str) -> Tuple[float, ...]:
    #     """A helper method to get default color for the series.
    #
    #     Args:
    #         name: Name of the series.
    #
    #     Returns:
    #         Default color available in matplotlib.
    #     """
    #     if name not in self._series:
    #         self._series.append(name)
    #
    #     ind = self._series.index(name) % len(self.DefaultColors)
    #     return self.DefaultColors[ind]
    #
    # def _get_default_marker(self, name: str) -> str:
    #     """A helper method to get default marker for the scatter plot.
    #
    #     Args:
    #         name: Name of the series.
    #
    #     Returns:
    #         Default marker available in matplotlib.
    #     """
    #     if name not in self._series:
    #         self._series.append(name)
    #
    #     ind = self._series.index(name) % len(self.DefaultMarkers)
    #     return self.DefaultMarkers[ind]

    def _update_label_in_options(
        self,
        options: Dict[str, any],
        name: Optional[str],
        label: Optional[str] = None,
        legend: bool = False,
    ):
        """Helper function to set the label entry in ``options`` based on given arguments.

        This method uses :meth:`label_for` to get the label for the series identified by ``name``. If
        :meth:`label_for` returns ``None``, then ``_update_label_in_options`` doesn't add a `"label"`
        entry into ``options``. I.e., a label entry is added to ``options`` only if it is not ``None``.

        Args:
            options: The options dictionary being modified.
            name: The name of the series being labelled. Used as a fall-back label if ``label`` is None
                and no label exists in ``series_params`` for this series.
            label: Optional legend label to override ``name`` and ``series_params``.
            legend: Whether a label entry should be added to ``options``. USed as an easy toggle to
                disable adding a label entry. Defaults to False.
        """
        if legend:
            _label = self.label_for(name, label)
            if _label:
                options["name"] = _label
        else:
            options["showlegend"] = False

    def draw_scatter(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        x_err: Optional[Sequence[float]] = None,
        y_err: Optional[Sequence[float]] = None,
        name: Optional[str] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):

        series_params = self.figure_options.series_params.get(name, {})
        # marker = series_params.get("symbol", self._get_default_marker(name))
        # color = series_params.get("color", self._get_default_color(name))
        axis = series_params.get("canvas", None)

        draw_options = {
            "x": x_data,
            "y": y_data,
            "mode": 'markers',
            "legendgroup": name,
            # "color": color,
            # "marker": marker,
            # "alpha": 0.8,
            # "zorder": 2,
        }
        self._update_label_in_options(draw_options, name, label, legend)
        # draw_options.update(**options)

        if y_err is not None and np.all(np.isfinite(y_err)):
            draw_options["error_y"] = {
                "type": "data",
                "array": y_err,
                "visible": True,
            }

        if x_err is not None and np.all(np.isfinite(x_err)):
            draw_options["error_x"] = {
                "type": "data",
                "array": x_err,
                "visible": True,
            }

        self._get_axis(axis).add_trace(go.Scatter(**draw_options))

    def draw_line(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        name: Optional[str] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):
        series_params = self.figure_options.series_params.get(name, {})
        axis = series_params.get("canvas", None)
        # color = series_params.get("color", self._get_default_color(name))

        draw_options = {
            "x": x_data,
            "y": y_data,
            "mode": 'lines',
            "legendgroup": name,
            "hoverinfo": "skip",
            # "color": color,
            # "linestyle": "-",
            # "linewidth": 2,
        }
        self._update_label_in_options(draw_options, name, label, legend)
        # draw_ops.update(**options)

        self._get_axis(axis).add_trace(go.Scatter(**draw_options))

    def draw_filled_y_area(
        self,
        x_data: Sequence[float],
        y_ub: Sequence[float],
        y_lb: Sequence[float],
        name: Optional[str] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):
        series_params = self.figure_options.series_params.get(name, {})
        axis = series_params.get("canvas", None)
        # color = series_params.get("color", self._get_default_color(name))

        # hack from https://community.plotly.com/t/fill-area-between-two-lines/26890/4
        xconcat = np.concatenate([x_data, x_data[::-1]])
        yconcat = np.concatenate([y_ub, y_lb[::-1]])
        alpha = options.get("alpha", 0.1)

        draw_ops = {
            # "alpha": 0.1,
            # "color": color,
            "x": xconcat,
            "y": yconcat,
            "fill": "toself",
            "mode": "none",
            "legendgroup": name,
            "hoverinfo": "skip",
            "fillcolor": f"rgba(0, 0, 255, {alpha})"  # hard-coded color now
        }
        self._update_label_in_options(draw_ops, name, label, legend)
        # draw_ops.update(**options)
        self._get_axis(axis).add_trace(go.Scatter(**draw_ops))

    def draw_filled_x_area(
        self,
        x_ub: Sequence[float],
        x_lb: Sequence[float],
        y_data: Sequence[float],
        name: Optional[str] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):
        series_params = self.figure_options.series_params.get(name, {})
        axis = series_params.get("canvas", None)
        # color = series_params.get("color", self._get_default_color(name))

        # hack from https://community.plotly.com/t/fill-area-between-two-lines/26890/4
        xconcat = np.concatenate([x_ub, x_lb[::-1]])
        yconcat = np.concatenate([y_data, y_data[::-1]])
        alpha = options.get("alpha", 0.1)

        draw_ops = {
            # "alpha": 0.1,
            # "color": color,
            "x": xconcat,
            "y": yconcat,
            "fill": "toself",
            "mode": "none",
            "legendgroup": name,
            "hoverinfo": "skip",
            "fillcolor": f"rgba(0, 0, 255, {alpha})"  # hard-coded color now
        }
        self._update_label_in_options(draw_ops, name, label, legend)
        # draw_ops.update(**options)
        self._get_axis(axis).add_trace(go.Scatter(**draw_ops))

    def draw_text_box(
        self,
        description: str,
        rel_pos: Optional[Tuple[float, float]] = None,
        **options,
    ):
        html_format = description.replace("\n", "<br>")

        if rel_pos is None:
            rel_pos = self.style["textbox_rel_pos"]

        text_options = {
            "text": html_format,
            "align": "left",
            "xref": "paper",
            "yref": "paper",
            "x": rel_pos[0],
            "y": rel_pos[1],
            "borderwidth": 1,
            "opacity": 0.85,
            "showarrow": False,
            "clicktoshow": "onout",
        }

        self._axis.add_annotation(**text_options)

        # bbox_props = {
        #     "boxstyle": "square, pad=0.3",
        #     "fc": "white",
        #     "ec": "black",
        #     "lw": 1,
        #     "alpha": 0.8,
        # }
        # bbox_props.update(**options)
        #
        # if rel_pos is None:
        #     rel_pos = self.style["textbox_rel_pos"]
        #
        # text_box_handler = self._axis.text(
        #     *rel_pos,
        #     s=description,
        #     ha="center",
        #     va="top",
        #     size=self.style["textbox_text_size"],
        #     transform=self._axis.transAxes,
        #     zorder=1000,  # Very large zorder to draw over other graphics.
        # )
        # text_box_handler.set_bbox(bbox_props)

    @property
    def figure(self) -> go.Figure:
        """Return figure object handler to be saved in the database."""
        return self._axis
