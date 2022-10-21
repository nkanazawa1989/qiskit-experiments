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

"""Experiment utility functions."""

import io
import logging
import threading
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Callable, Tuple, List, Dict, Any, Union, Type, Optional
import json

import dateutil.parser
import pkg_resources
from dateutil import tz
from qiskit.version import __version__ as terra_version
import pandas as pd

try:
    from qiskit.providers.ibmq.experiment import (
        IBMExperimentEntryExists,
        IBMExperimentEntryNotFound,
    )

    HAS_IBMQ = True
except ImportError:
    HAS_IBMQ = False

from .exceptions import ExperimentEntryNotFound, ExperimentEntryExists, ExperimentDataError
from ..version import __version__ as experiments_version

LOG = logging.getLogger(__name__)


def qiskit_version():
    """Return the Qiskit version."""
    try:
        return pkg_resources.get_distribution("qiskit").version
    except Exception:  # pylint: disable=broad-except
        return {"qiskit-terra": terra_version, "qiskit-experiments": experiments_version}


def parse_timestamp(utc_dt: Union[datetime, str]) -> datetime:
    """Parse a UTC ``datetime`` object or string.

    Args:
        utc_dt: Input UTC `datetime` or string.

    Returns:
        A ``datetime`` with the UTC timezone.

    Raises:
        TypeError: If the input parameter value is not valid.
    """
    if isinstance(utc_dt, str):
        utc_dt = dateutil.parser.parse(utc_dt)
    if not isinstance(utc_dt, datetime):
        raise TypeError("Input `utc_dt` is not string or datetime.")
    utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    return utc_dt


def utc_to_local(utc_dt: datetime) -> datetime:
    """Convert input UTC timestamp to local timezone.

    Args:
        utc_dt: Input UTC timestamp.

    Returns:
        A ``datetime`` with the local timezone.
    """
    local_dt = utc_dt.astimezone(tz.tzlocal())
    return local_dt


def plot_to_svg_bytes(figure: "pyplot.Figure") -> bytes:
    """Convert a pyplot Figure to SVG in bytes.

    Args:
        figure: Figure to be converted

    Returns:
        Figure in bytes.
    """
    buf = io.BytesIO()
    opaque_color = list(figure.get_facecolor())
    opaque_color[3] = 1.0  # set alpha to opaque
    figure.savefig(
        buf, format="svg", facecolor=tuple(opaque_color), edgecolor="none", bbox_inches="tight"
    )
    buf.seek(0)
    figure_data = buf.read()
    buf.close()
    return figure_data


def save_data(
    is_new: bool,
    new_func: Callable,
    update_func: Callable,
    new_data: Dict,
    update_data: Dict,
    json_encoder: Optional[Type[json.JSONEncoder]] = None,
) -> Tuple[bool, Any]:
    """Save data in the database.

    Args:
        is_new: ``True`` if `new_func` should be called. Otherwise `update_func` is called.
        new_func: Function to create new entry in the database.
        update_func: Function to update an existing entry in the database.
        new_data: In addition to `update_data`, this data will be stored if creating
            a new entry.
        update_data: Data to be stored if updating an existing entry.
        json_encoder: Custom JSON encoder to use to encode the experiment.

    Returns:
        A tuple of whether the data was saved and the function return value.

    Raises:
        ExperimentDataError: If unable to determine whether the entry exists.
    """
    attempts = 0
    no_entry_exception = [ExperimentEntryNotFound]
    dup_entry_exception = [ExperimentEntryExists]
    if HAS_IBMQ:
        no_entry_exception.append(IBMExperimentEntryNotFound)
        dup_entry_exception.append(IBMExperimentEntryExists)
    try:
        kwargs = {}
        if json_encoder:
            kwargs["json_encoder"] = json_encoder
        # Attempt 3x for the unlikely scenario wherein is_new=False but the
        # entry doesn't actually exist. The second try might also fail if an entry
        # with the same ID somehow got created in the meantime.
        while attempts < 3:
            attempts += 1
            if is_new:
                try:
                    kwargs.update(new_data)
                    kwargs.update(update_data)
                    return True, new_func(**kwargs)
                except tuple(dup_entry_exception):
                    is_new = False
            else:
                try:
                    kwargs.update(update_data)
                    return True, update_func(**kwargs)
                except tuple(no_entry_exception):
                    is_new = True
        raise ExperimentDataError("Unable to determine the existence of the entry.")
    except Exception:  # pylint: disable=broad-except
        # Don't fail the experiment just because its data cannot be saved.
        LOG.error("Unable to save the experiment data: %s", traceback.format_exc())
        return False, None


class ThreadSafeContainer(ABC):
    """Base class for thread safe container."""

    def __init__(self, init_values=None):
        """ThreadSafeContainer constructor."""
        self._lock = threading.RLock()
        self._container = self._init_container(init_values)

    @abstractmethod
    def _init_container(self, init_values):
        """Initialize the container."""
        pass

    def __iter__(self):
        with self._lock:
            return iter(self._container)

    def __getitem__(self, key):
        with self._lock:
            return self._container[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._container[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._container[key]

    def __contains__(self, item):
        with self._lock:
            return item in self._container

    def __len__(self):
        with self._lock:
            return len(self._container)

    @property
    def lock(self):
        """Return lock used for this container."""
        return self._lock

    def copy(self):
        """Returns a copy of the container."""
        with self.lock:
            return self._container.copy()

    def copy_object(self):
        """Returns a copy of this object."""
        obj = self.__class__()
        obj._container = self.copy()
        return obj

    def clear(self):
        """Remove all elements from this container."""
        with self.lock:
            self._container.clear()

    def __json_encode__(self):
        cpy = self.copy_object()
        return {"_container": cpy._container}

    @classmethod
    def __json_decode__(cls, value):
        ret = cls()
        ret._container = value["_container"]
        return ret

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove non-pickleable attribute
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Initialize non-pickleable attribute
        self._lock = threading.RLock()


class ThreadSafeOrderedDict(ThreadSafeContainer):
    """Thread safe OrderedDict."""

    def _init_container(self, init_values):
        """Initialize the container."""
        return OrderedDict.fromkeys(init_values or [])

    def get(self, key, default):
        """Return the value of the given key."""
        with self._lock:
            return self._container.get(key, default)

    def keys(self):
        """Return all key values."""
        with self._lock:
            return list(self._container.keys())

    def values(self):
        """Return all values."""
        with self._lock:
            return list(self._container.values())

    def items(self):
        """Return the key value pairs."""
        return self._container.items()


class ThreadSafeList(ThreadSafeContainer):
    """Thread safe list."""

    def _init_container(self, init_values):
        """Initialize the container."""
        return init_values or []

    def append(self, value):
        """Append to the list."""
        with self._lock:
            self._container.append(value)


class ThreadSafeDataFrame(ThreadSafeContainer):
    """Thread safe data frame base class."""

    def __init__(self, init_values=None):
        """ThreadSafeContainer constructor."""
        self._columns = self._default_columns()
        super().__init__(init_values)

    @classmethod
    def _default_columns(cls) -> List[str]:
        return []

    def _init_container(self, init_values: Optional[Union[Dict, List]] = None):
        """Initialize the container."""
        if init_values is None:
            return pd.DataFrame(columns=self.get_columns())
        if isinstance(init_values, list):
            if not isinstance(init_values[0], list):
                # This must be nested list. Each element corresponds to a single row.
                init_values = [init_values]
            formatted = {}
            for init_value in init_values:
                # generate unique entry id.
                entry_id = str(uuid.uuid4())
                formatted[entry_id] = init_value
        else:
            formatted = init_values
        if isinstance(formatted, dict):
            return pd.DataFrame.from_dict(
                data=formatted,
                orient="index",
                columns=self.get_columns(),
            )
        raise TypeError(
            f"Invalid data type '{init_values.__class__.__name__}'."
        )

    def get_columns(self) -> List[str]:
        """Return current column names.

        Returns:
            List of column names.
        """
        return self._columns.copy()

    def add_columns(self, *new_columns: str):
        """Add new columns to the table.

        This operation mutates the current container.

        Args:
            new_columns: Name of columns to add.
        """
        # Order sensitive
        new_columns = [c for c in new_columns if c not in self.get_columns()]
        # Update current table
        with self._lock:
            for new_column in new_columns:
                self._container.insert(len(self._container.columns), new_column, None)
        self._columns.extend(new_columns)

    def clear(self):
        """Remove all elements from this container."""
        with self._lock:
            self._container = self._init_container()

    def truncated_table(self) -> pd.DataFrame:
        """Return dataframe with truncated columns.

        Columns with underscored name are removed for visualization for the end-users.

        Returns:
            Data frame.
        """
        to_show = [c for c in self.get_columns() if not c.startswith("_")]
        with self._lock:
            return self._container[to_show]

    def full_table(self) -> pd.DataFrame:
        """Return dataframe with full columns.

        Returns:
            Data frame.
        """
        return self.copy()

    def _repr_html_(self) -> Union[str, None]:
        with self._lock:
            return self.truncated_table()._repr_html_()

    def add_data(self, entry_id: Optional[str] = None, **kwargs):
        """Add new data.

        Args:
            entry_id: Unique name of this entry.
            kwargs: Data of new entry to register. Key must agree with column names.

        Raises:
            KeyError: When invalid key is specified.
        """
        columns = self.get_columns()
        if not set(columns).issuperset(kwargs.keys()):
            missing = ", ".join(kwargs.keys() - set(columns))
            raise KeyError(f"Invalid keys {missing} are found. These keys are not defined.")
        template = dict.fromkeys(columns)
        template.update(kwargs)
        entry_id = entry_id or str(uuid.uuid4())
        with self._lock:
            self._container.loc[entry_id] = list(template.values())

    def __getattr__(self, item):
        lock = object.__getattribute__(self, "_lock")

        with lock:
            # Lock when access to container's member.
            container = object.__getattribute__(self, "_container")
            if hasattr(container, item):
                return getattr(container, item)
        raise AttributeError(f"'ThreadSafeDataFrame' object has no attribute '{item}'")


class AnalysisResultTable(ThreadSafeDataFrame):

    @classmethod
    def _default_columns(cls) -> List[str]:
        return [
            "name",
            "value",
            "chisq",
            "quality",
            "device_components",
            "_extra",
            "_experiment_id",
            "_verified",
            "_tags",
            "_source",
            "_created_in_db",
        ]
