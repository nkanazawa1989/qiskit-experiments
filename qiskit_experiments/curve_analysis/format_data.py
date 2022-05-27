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
Functions to perform fit data formatting.
"""

from typing import Dict, Any

import pandas as pd
import numpy as np


def shot_weighted_average(group: pd.DataFrame):
    """Take average over the same x values. Performs average weighted by shots.

    Args:
        group: Dataframe grouped by the model and x values.

    Returns:
        Average of the group.
    """
    if len(group) == 1:
        return group.sum()

    total_shots = np.sum(group.shots)
    weights = group.shots / total_shots

    out = {
        "x": group.x.iloc[0],
        "y": np.sum(weights * group.y),
        "y_err": np.sqrt(np.sum(weights ** 2 * group.y_err ** 2)),
        "shots": total_shots,
    }
    _add_extra(group, out)

    return pd.Series(data=out.values(), index=out.keys())


def sampled_average(group: pd.DataFrame):
    """Take average over the same x values. Performs sampled average.

    .. notes::

        Original error bar of y data is discarded.
        Error bars are newly computed as a standard deviation of multiple y values.

    Args:
        group: Dataframe grouped by the model and x values.

    Returns:
        Average of the group.
    """
    if len(group) == 1:
        return group.sum()

    total_shots = np.sum(group.shots)
    y_mean = np.mean(group.y)

    out = {
        "x": group.x.iloc[0],
        "y": y_mean,
        "y_err": np.sqrt(np.mean(y_mean - group.y) ** 2 / len(group)),
        "shots": total_shots,
    }
    _add_extra(group, out)

    return pd.Series(data=out.values(), index=out.keys())


def _add_extra(group: pd.DataFrame, series: Dict[str, Any]):
    """A helper function to add extra columns.

    Only unique values are added to the returned series.
    If a colum contains multiple value, then the column values are discarded.
    This mutably updates input series dictionary.

    Args:
        group: DataFrame.
        series: Dictionary that is a merger of duplicated x values.
    """
    for k, v in group.items():
        if k not in series:
            if len(set(v)) == 1:
                series[k] = v.iloc[0]
            else:
                series[k] = pd.NA
