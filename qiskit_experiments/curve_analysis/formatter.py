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
Helper functions to average experiment data.
"""

import pandas as pd
import numpy as np


AVGFUN_COL_NAMES = [
    "y_val",
    "y_err",
    "samples",
    "model_name",
    "analysis_group",
    "data_kind",
]


def shot_weighted_average(group: pd.DataFrame) -> pd.Series:
    """Shot weighted average over all rows.

    Args:
        group: Data frame grouped bye the model and x values.

    Returns:
        Average of the group.
    """
    if len(group) == 1:
        return group.sum()

    total_shots = group.shots.sum()
    weights = group.shots / total_shots

    return pd.Series(
        data=[
            np.sum(weights * group.y_val),
            np.sqrt(np.sum(weights**2 * group.y_err**2)),
            total_shots,
            group.model_name.iloc[0],
            group.analysis_group.iloc[0],
            "formatted",
        ],
        index=AVGFUN_COL_NAMES,
    )


def sample_average(group: pd.DataFrame) -> pd.Series:
    """Sample average over all rows.

    Args:
        group: Data frame grouped bye the model and x values.

    Returns:
        Average of the group.
    """
    if len(group) == 1:
        return group.sum()

    y_mean = group.y_val.mean()

    return pd.Series(
        data=[
            y_mean,
            np.sqrt(np.mean((y_mean - group.y_val)**2) / len(group)),
            len(group),
            group.model_name.iloc[0],
            group.analysis_group.iloc[0],
            "formatted",
        ],
        index=AVGFUN_COL_NAMES,
    )


def iwv_average(group: pd.DataFrame) -> pd.Series:
    """Sample average over all rows.

    Args:
        group: Data frame grouped bye the model and x values.

    Returns:
        Average of the group.
    """
    raise NotImplementedError
