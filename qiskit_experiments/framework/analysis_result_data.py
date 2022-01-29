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

"""Helper dataclass for constructing analysis results."""

import logging
import warnings
from typing import Optional, Dict, Any, List

from uncertainties import UFloat, ufloat

from qiskit_experiments.database_service import DbAnalysisResultV1
from qiskit_experiments.database_service.db_fitval import FitVal
from qiskit_experiments.database_service.device_component import DeviceComponent, to_component
from qiskit_experiments.database_service.exceptions import DbExperimentDataError

LOG = logging.getLogger(__name__)


class AnalysisResult(DbAnalysisResultV1):
    """Qiskit Experiments Analysis Result container class.

    This object is intended to be used for storing result of analysis.
    Thus class can be instantiated without experiment metadata nor provider information.
    Missing information can be set later if user want to save the data to database.

    This class also supports expression of the analysis value with the UFloat object.
    UFloat objects are implicitly converted into :class:`FitVal`
    for serialization when the entry is saved in the database.
    """
    # TODO remove provider API from this class. This should be simple container for analysis.

    def __init__(
        self,
        name: str,
        value: Any,
        unit: Optional[str] = None,
        chisq: Optional[float] = None,
        quality: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        device_components: Optional[List[DeviceComponent]] = None,
    ):
        """Create new result object.

        Args:
            name: Name of the quantity saved in this container.
            value: Value from the analysis.
            unit: Physical unit of the value if exist.
            chisq: Reduced Chi-squared value if exist.
            quality: Quality of analysis.
            extra: Metadata associated to the analysis.
            device_components: Target device components for this analysis result
        """
        super().__init__(
            name=name,
            value=value,
            device_components=device_components or list(),
            experiment_id="",
            chisq=chisq,
            quality=quality,
            extra=extra,
        )
        self._unit = unit

    @property
    def experiment_id(self) -> str:
        """Return the ID of the experiment associated with this analysis result.

        Returns:
            ID of experiment associated with this analysis result.
        """
        return self._experiment_id

    @experiment_id.setter
    def experiment_id(self, new_id: str):
        """Set new experiment id if not exist.

        Args:
            new_id: ID of experiment associated with this analysis result.
        """
        if not self.experiment_id:
            self._experiment_id = new_id
        else:
            warnings.warn(
                "Experiment ID cannot be overridden. Create new object to set new value.",
                UserWarning,
            )

    @property
    def device_components(self) -> List[DeviceComponent]:
        """Return target device components for this analysis result.

        Returns:
            Target device components.
        """
        return self._device_components

    @device_components.setter
    def device_components(self, new_components: List[DeviceComponent]):
        """Set new target device components if not exist.

        Args:
            new_components: Target device components.
        """
        if not self.device_components:
            self._device_components = [
                to_component(comp) if isinstance(comp, str) else comp for comp in new_components
            ]
        else:
            warnings.warn(
                "Device components cannot be overridden. Create new object to set new value.",
                UserWarning,
            )

    @property
    def unit(self):
        """Return physical unit of value stored in this container.

        Returns:
            Physical unit of analysis value.
        """
        return self._unit

    @unit.setter
    def unit(self, new_unit: str):
        """Set new unit.

        Args:
            new_unit: Physical unit of analysis value.
        """
        self._unit = new_unit

    def save(self):
        """Save this analysis result in the database.

        Raises:
            DbExperimentDataError: When the experiment metadata is not set.
        """
        if isinstance(self.value, UFloat):
            db_value = FitVal(
                value=self.value.nominal_value,
                stderr=self.value.std_dev,
                unit=self._unit,
            )
        else:
            if self.unit:
                db_value = FitVal(value=self.value, unit=self.unit)
            else:
                db_value = self.value

        if not self.experiment_id or not self.device_components:
            raise DbExperimentDataError(
                "Required fields are missing. "
                "Set self.experiment_id and self.device_components from the experiment."
            )

        db_analysis_result = DbAnalysisResultV1(
            name=self.name,
            value=db_value,
            device_components=self.device_components,
            experiment_id=self.experiment_id,
            result_id=self.result_id,
            chisq=self.chisq,
            quality=self.quality,
            extra=self.extra,
            verified=self.verified,
            tags=self.tags,
            service=self.service,
            source=self.source,
        )
        db_analysis_result.save()

    @classmethod
    def _from_service_data(cls, service_data: Dict) -> "AnalysisResult":
        """Construct an analysis result from saved database service data.

        Args:
            service_data: Analysis result data.

        Returns:
            The loaded analysis result.
        """
        result_data = service_data.pop("result_data")
        db_value = result_data.pop("_value")
        chisq = result_data.pop("_chisq", None)
        extra = result_data.pop("_extra", {})
        source = result_data.pop("_source", None)

        if isinstance(db_value, FitVal):
            value = ufloat(db_value.value, db_value.stderr)
            unit = db_value.unit
        else:
            value = db_value
            unit = None

        obj = cls(
            name=service_data.pop("result_type"),
            value=value,
            unit=unit,
            chisq=chisq,
            quality=service_data.pop("quality"),
            extra=extra,
        )

        # private properties
        obj._id = service_data.pop("result_id")
        obj._source = source

        # with setters
        obj.experiment_id = service_data.pop("experiment_id")
        obj.device_components = service_data.pop("device_components")
        obj.tags = service_data.pop("tags")
        obj.verified = service_data.pop("verified")
        obj.service = service_data.pop("service")

        # TODO this should not exist.
        #  We should be able to define __slots__ for memory efficiency
        for key, val in service_data.items():
            setattr(obj, key, val)
        return obj

    def copy(self) -> "AnalysisResult":
        """Return a copy of the result with a new result ID"""
        new_obj = AnalysisResult(
            name=self.name,
            value=self.value,
            unit=self.unit,
            chisq=self.chisq,
            quality=self.quality,
            extra=self.extra.copy(),
        )
        new_obj.experiment_id = self.experiment_id
        new_obj.device_components = self.device_components
        new_obj.tags = self.tags
        new_obj.verified = self.verified
        new_obj.service = self.service

        new_obj._id = self.result_id
        new_obj._source = self.source

        return new_obj

    def __str__(self):
        ret = f"{type(self).__name__}"
        ret += f"\n- name: {self.name}"
        ret += f"\n- value: {str(self.value)}"
        if self.unit is not None:
            ret += f" [{self.unit}]"
        if self.chisq is not None:
            ret += f"\n- χ²: {str(self.chisq)}"
        if self.quality is not None:
            ret += f"\n- quality: {self.quality}"
        if self.extra:
            ret += f"\n- extra: <{len(self.extra)} items>"
        ret += f"\n- device_components: {[str(i) for i in self.device_components]}"
        ret += f"\n- verified: {self.verified}"
        return ret

    def __repr__(self):
        out = f"{type(self).__name__}("
        out += f"name={self.name}"
        out += f", value={repr(self.value)}"
        out += f", unit={self.unit}"
        out += f", device_components={repr(self.device_components)}"
        out += f", experiment_id={self.experiment_id}"
        out += f", result_id={self.result_id}"
        out += f", chisq={self.chisq}"
        out += f", quality={self.quality}"
        out += f", verified={self.verified}"
        out += f", extra={repr(self.extra)}"
        out += f", tags={self.tags}"
        out += f", service={repr(self.experiment_id)}"
        for key, val in self._extra_data.items():
            out += f", {key}={repr(val)}"
        out += ")"
        return out


class AnalysisResultData(AnalysisResult):
    """Deprecated. A container for experiment analysis results"""

    def __new__(
        cls,
        name: str,
        value: Any,
        chisq: Optional[float] = None,
        quality: Optional[str] = None,
        extra: Optional[Dict[str, str]] = None,
        device_components: Optional[List[DeviceComponent]] = None,
    ):
        """Instantiate new AnalysisResult class. This class is deprecated.

        Args:
            name: Name of the quantity saved in this container.
            value: Value from the analysis.
            chisq: Reduced Chi-squared value if exist.
            quality: Quality of analysis.
            extra: Metadata associated to the analysis.
            device_components: Target device components.
        """
        warnings.warn(
            "AnalysisResultData has been deprecated in Qiskit Experiments 0.3 and "
            "will be removed in 0.4 release. Use AnalysisResult class instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        instance = AnalysisResult(
            name=name,
            value=value,
            chisq=chisq,
            quality=quality,
            extra=extra,
            device_components=device_components,
        )

        return instance
