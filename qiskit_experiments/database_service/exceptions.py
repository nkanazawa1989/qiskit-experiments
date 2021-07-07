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

"""Exceptions for errors raised by experiment service data."""

from qiskit.exceptions import QiskitError


class DbExperimentDataError(QiskitError):
    """Base class for errors raised by experiment service data."""

    pass


class DbExperimentEntryNotFound(DbExperimentDataError):
    """Errors raised when an experiment entry cannot be found."""

    pass


class DbExperimentEntryExists(DbExperimentDataError):
    """Errors raised when an experiment entry already exists."""

    pass
