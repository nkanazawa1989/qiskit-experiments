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


def shot_weighted_average(df: pd.DataFrame):

    total_shots = np.sum(df.shots)
    weights = df.shots / total_shots

    out = {
        "x": df.x.iloc[0],
        "y": np.sum(weights * df.y),
        "y_err": np.sqrt(np.sum(weights**2 * df.y_err**2)),
        "shots": total_shots,
    }
    _add_extra(df, out)

    return pd.Series(data=out.values(), index=out.keys())


def sampled_average(df: pd.DataFrame):

    total_shots = np.sum(df.shots)
    y_mean = np.mean(df.y)

    out = {
        "x": df.x.iloc[0],
        "y": y_mean,
        "y_err": np.sqrt(np.mean(y_mean - df.y)**2 / len(df)),
        "shots": total_shots,
    }
    _add_extra(df, out)

    return pd.Series(data=out.values(), index=out.keys())


def _add_extra(df: pd.DataFrame, series: Dict[str, Any]):
    """A helper function to add extra columns.

    Only unique values are added to the returned series.
    If a colum contains multiple value, then the column values are discarded.
    This mutably updates input series dictionary.

    Args:
        df: DataFrame.
        series: Dictionary that is a merger of duplicated x values.
    """
    for k, v in df.items():
        if k not in series:
            if len(set(v)) == 1:
                series[k] = v.iloc[0]
            else:
                series[k] = pd.NA
