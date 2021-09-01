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
T2Ramsey Experiment class.

"""

from typing import List, Optional, Union
import numpy as np

import qiskit
from qiskit.utils import apply_prefix
from qiskit.providers import Backend
from qiskit.test.mock import FakeBackend
from qiskit.circuit import QuantumCircuit
from qiskit.providers.options import Options
from qiskit_experiments.framework import BaseExperiment
from .t2ramsey_analysis import T2RamseyAnalysis


class T2Ramsey(BaseExperiment):

    r"""
    T2Ramsey Experiment

    Overview
        This experiment is used to estimate two properties for a single qubit:
        T2* and Ramsey frequency.
        This experiment consists of a series of circuits of the form

    .. parsed-literal::

           ┌───┐┌──────────────┐┌──────┐ ░ ┌───┐ ░ ┌─┐
      q_0: ┤ H ├┤   DELAY(t)   ├┤ RZ(λ)├─░─┤ H ├─░─┤M├
           └───┘└──────────────┘└──────┘ ░ └───┘ ░ └╥┘
      c: 1/═════════════════════════════════════════╩═
                                                    0

    for each *t* from the specified delay times, and where
    :math:`\lambda =2 \pi \times {osc\_freq}`,
    and the delays are specified by the user.
    The circuits are run on the device or on a simulator backend.
    Results are analysed in the class T2RamseyAnalysis.

    Analysis Class
        :class:`~qiskit.experiments.characterization.T2RamseyAnalysis`

    Analysis Options

        - **user_p0** (``List[Float]``): user guesses for the fit parameters
          ``A``, ``B``, ``f``, ``phi``, ``T2star``.
        - **bounds** - (Tuple[List[float], List[float]]) lower and upper bounds for the fit parameters.
        - **plot** (bool) - create a graph if and only if True.

    """
    __analysis_class__ = T2RamseyAnalysis

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            delays (Iterable[float]): Delay times of the experiments.
            unit (str): Unit of the delay times. Supported units are
                's', 'ms', 'us', 'ns', 'ps', 'dt'.
            osc_freq (float): Oscillation frequency offset in Hz.
        """

        return Options(delays=None, unit="s", osc_freq=0.0)

    def __init__(
        self,
        qubit: int,
        delays: Union[List[float], np.array],
        unit: str = "s",
        osc_freq: float = 0.0,
    ):
        """
        **T2Ramsey class**

        Initialize the T2Ramsey class.

        Args:
            qubit: the qubit under test.
            delays: delay times of the experiments.
            unit: Optional, time unit of `delays`.
            Supported units: 's', 'ms', 'us', 'ns', 'ps', 'dt'. The unit is \
            used for both T2Ramsey and for the frequency.
            osc_freq: optional, oscillation frequency offset in Hz.
        """

        super().__init__([qubit])
        self.set_experiment_options(delays=delays, unit=unit, osc_freq=osc_freq)

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Each circuit consists of a Hadamard gate, followed by a fixed delay,
        a phase gate (with a linear phase), and an additional Hadamard gate.

        Args:
            backend: Optional, a backend object

        Returns:
            The experiment circuits

        Raises:
            AttributeError: if unit is 'dt', but 'dt' parameter is missing in the backend configuration.
        """
        conversion_factor = 1
        if self.experiment_options.unit == "dt":
            try:
                dt_factor = getattr(backend._configuration, "dt")
                conversion_factor = dt_factor
            except AttributeError as no_dt:
                raise AttributeError("Dt parameter is missing in backend configuration") from no_dt
        elif self.experiment_options.unit != "s":
            apply_prefix(1, self.experiment_options.unit)

        circuits = []
        for delay in self.experiment_options.delays:
            circ = qiskit.QuantumCircuit(1, 1)
            circ.h(0)
            circ.delay(delay, 0, self.experiment_options.unit)
            rotation_angle = (
                2 * np.pi * self.experiment_options.osc_freq * conversion_factor * delay
            )
            circ.rz(rotation_angle, 0)
            circ.barrier(0)
            circ.h(0)
            circ.barrier(0)
            circ.measure(0, 0)

            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "osc_freq": self.experiment_options.osc_freq,
                "xval": delay,
                "unit": self.experiment_options.unit,
            }
            if self.experiment_options.unit == "dt":
                circ.metadata["dt_factor"] = dt_factor

            circuits.append(circ)

        return circuits

    def _pre_transpile_hook(self, backend: Backend):
        """Set timing constraints if backend is real hardware."""

        if not backend.configuration().simulator and not isinstance(backend, FakeBackend):
            timing_constraints = getattr(self.transpile_options.__dict__, "timing_constraints", {})

            # alignment=16 is IBM standard. Will be soon provided by IBM providers.
            # Then, this configuration can be removed.
            timing_constraints["acquire_alignment"] = getattr(
                timing_constraints, "acquire_alignment", 16
            )

            scheduling_method = getattr(
                self.transpile_options.__dict__, "scheduling_method", "alap"
            )
            self.set_transpile_options(
                timing_constraints=timing_constraints, scheduling_method=scheduling_method
            )
