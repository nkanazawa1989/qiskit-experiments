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

"""Test the half angle experiment."""

from test.base import QiskitExperimentsTestCase
import copy
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.test.mock import FakeAthens
import qiskit.pulse as pulse
from qiskit.pulse import InstructionScheduleMap

from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.library import HalfAngle


class HalfAngleTestBackend(MockIQBackend):
    """A simple and primitive backend, to be run by the half angle tests."""

    def __init__(self, error: float):
        """Initialize the class."""
        super().__init__()
        self._error = error

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Returns the probability of measuring the excited state."""

        n_gates = circuit.metadata["xval"]

        return 0.5 * np.sin((-1) ** (n_gates + 1) * n_gates * self._error) + 0.5


class TestHalfAngle(QiskitExperimentsTestCase):
    """Class to test the half angle experiment."""

    def test_end_to_end(self):
        """Test a full experiment end to end."""

        tol = 0.005
        for error in [-0.05, -0.02, 0.02, 0.05]:
            hac = HalfAngle(0)
            exp_data = hac.run(HalfAngleTestBackend(error))
            d_theta = exp_data.analysis_results(1).value.value

            self.assertTrue(abs(d_theta - error) < tol)

    def test_circuits(self):
        """Test that transpiling works and that we can have a y gate with a calibration."""

        qubit = 1

        inst_map = InstructionScheduleMap()
        for inst in ["sx", "y"]:
            inst_map.add(inst, (qubit,), pulse.Schedule(name=inst))

        hac = HalfAngle(qubit)
        hac.set_transpile_options(inst_map=inst_map)

        # mimic what will happen in the experiment.
        transpile_opts = copy.copy(hac.transpile_options.__dict__)
        transpile_opts["initial_layout"] = list(hac._physical_qubits)
        circuits = transpile(hac.circuits(), FakeAthens(), **transpile_opts)

        for idx, circ in enumerate(circuits):
            self.assertEqual(circ.count_ops()["sx"], idx * 2 + 2)
            self.assertEqual(circ.calibrations["sx"][((qubit,), ())], pulse.Schedule(name="sx"))
            if idx > 0:
                self.assertEqual(circ.count_ops()["y"], idx)
                self.assertEqual(circ.calibrations["y"][((qubit,), ())], pulse.Schedule(name="y"))

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = HalfAngle(1)
        config = exp.config
        loaded_exp = HalfAngle.from_config(config)
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqual(config, loaded_exp.config)
