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
Cross resonance Hamiltonian tomography.
"""

from typing import List, Tuple, Iterable, Dict, Optional

import numpy as np
import itertools
from qiskit import pulse, circuit, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
from qiskit.utils import optionals
from qiskit_experiments.framework import BaseExperiment, BatchExperiment, Options
from qiskit_experiments.curve_analysis import ParameterRepr

from .analysis.cr_hamiltonian_analysis import (
    TomographyElementAnalysis,
    CrossResonanceHamiltonianAnalysis,
)


class TomographyElement(BaseExperiment):

    def __init__(
        self,
        qubits: Tuple[int, int],
        tomography_circuit: QuantumCircuit,
        backend: Optional[Backend] = None,
    ):
        super().__init__(qubits=qubits, backend=backend, analysis=TomographyElementAnalysis())
        self.tomography_circuit = tomography_circuit
        self.param_map_r = {param.name: param for param in tomography_circuit.parameters}

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            pulse_parameters (Dict[str, float]): Pulse parameters keyed on the
                name of parameters attached to the ``tomography_circuit``.
            durations (Sequence[int]): ######
            xval_offset (float): Initial guess of xvalue offset due to
                rising and falling pulse edges. This should be provided by the
                root experiment, since thie experiment is agnostic to the pulse shape.
            t_risefall (float): Actual duration of pulse rising and falling edges.
                This should be provided by the root experiment,
                since thie experiment is agnostic to the pulse shape.
            dt (float): Time resoluton of the system. This parameter is
                automatically set when backend is provided.
            granularity (int): Constaints of pulse data chunk size. This parameter is
                automatically set when backend is provided.
        """
        options = super()._default_experiment_options()
        options.durations = None
        options.pulse_parameters = dict()
        options.xval_offset = 0
        options.t_risefall = 0
        options.dt = 1
        options.granularity = 1
        options.cr_channel = 0

        return options

    def _set_backend(self, backend):
        """Extract dt and granularity from the backend."""
        super()._set_backend(backend)
        configuration = backend.configuration()

        try:
            dt_factor = configuration.dt
        except AttributeError as ex:
            raise AttributeError(
                "Backend configuration does not provide system time resolution dt."
            ) from ex

        try:
            cr_channels = configuration.control(self.physical_qubits)
            index = cr_channels[0].index
        except AttributeError as ex:
            raise AttributeError(
                "Backend configuration does not provide control channel mapping."
            ) from ex

        try:
            granularity = configuration.timing_constraints["granularity"]
        except (AttributeError, KeyError):
            granularity = 1

        # Update experiment options
        self.set_experiment_options(dt=dt_factor, granularity=granularity, cr_channel=index)

    def set_experiment_options(self, **fields):
        """Set the experiment options.

        Args:
            fields: The fields to update the options
        """
        super().set_experiment_options(**fields)

        # Set initial guess of xval offset from the given pulse shapes
        xval_offset = self.experiment_options.xval_offset
        dt = self.experiment_options.dt
        self.analysis.set_options(p0={"t_off": xval_offset * dt})

    def circuits(self) -> List[QuantumCircuit]:
        opt = self.experiment_options

        tomo_circuits = []
        for meas_basis in ("x", "y", "z"):
            tomo_circ = QuantumCircuit(2, 1)

            tomo_circ.compose(
                other=self.tomography_circuit,
                qubits=[0, 1],
                inplace=True,
            )

            # measure
            if meas_basis == "x":
                tomo_circ.h(1)
            elif meas_basis == "y":
                tomo_circ.sdg(1)
                tomo_circ.h(1)
            tomo_circ.measure(1, 0)

            tomo_circ.metadata = {
                "experiment_type": self.experiment_type,
                "qubits": self.physical_qubits,
                "meas_basis": meas_basis,
            }
            tomo_circuits.append(tomo_circ)

        pulse_shape = {
            pobj: opt.pulse_parameters.get(pname, None) for pname, pobj in self.param_map_r.items()
        }
        pulse_shape[self.param_map_r["cr_channel"]] = opt.cr_channel

        experiment_circs = []
        for duration in opt.durations:
            effective_duration = opt.granularity * int(duration / opt.granularity)

            params = pulse_shape.copy()
            params[self.param_map_r["duration"]] = effective_duration

            for tomo_circ in tomo_circuits:
                tomo_circ_t = tomo_circ.assign_parameters(params)
                tomo_circ_t.metadata["xval"] = effective_duration * opt.dt  # in units of sec
                tomo_circ_t.metadata["pulse_shape"] = {p.name: v for p, v in params.items()}
                experiment_circs.append(tomo_circ_t)

        return experiment_circs


class CrossResonanceHamiltonian(BatchExperiment):
    r"""Cross resonance Hamiltonian tomography experiment.

    # section: overview

        This experiment assumes the two qubit Hamiltonian in the form

        .. math::

            H = \frac{I \otimes A}{2} + \frac{Z \otimes B}{2}

        where :math:`A` and :math:`B` are linear combinations of
        the Pauli operators :math:`\in {X, Y, Z}`.
        The coefficient of each Pauli term in the Hamiltonian
        can be estimated with this experiment.

        This experiment is performed by stretching the pulse duration of a cross resonance pulse
        and measuring the target qubit by projecting onto the x, y, and z bases.
        The control qubit state dependent (controlled-) Rabi oscillation on the
        target qubit is observed by repeating the experiment with the control qubit
        both in the ground and excited states. The fit for the oscillations in the
        three bases with the two control qubit preparations tomographically
        reconstructs the Hamiltonian in the form shown above.
        See Ref. [1] for more details.

        More specifically, the following circuits are executed in this experiment.

        .. parsed-literal::

            (X measurement)

                 ┌───┐┌────────────────────┐
            q_0: ┤ P ├┤0                   ├────────
                 └───┘│  cr_tone(duration) │┌───┐┌─┐
            q_1: ─────┤1                   ├┤ H ├┤M├
                      └────────────────────┘└───┘└╥┘
            c: 1/═════════════════════════════════╩═
                                                  0

            (Y measurement)

                 ┌───┐┌────────────────────┐
            q_0: ┤ P ├┤0                   ├───────────────
                 └───┘│  cr_tone(duration) │┌─────┐┌───┐┌─┐
            q_1: ─────┤1                   ├┤ Sdg ├┤ H ├┤M├
                      └────────────────────┘└─────┘└───┘└╥┘
            c: 1/════════════════════════════════════════╩═
                                                         0

            (Z measurement)

                 ┌───┐┌────────────────────┐
            q_0: ┤ P ├┤0                   ├───
                 └───┘│  cr_tone(duration) │┌─┐
            q_1: ─────┤1                   ├┤M├
                      └────────────────────┘└╥┘
            c: 1/════════════════════════════╩═
                                             0

        The ``P`` gate on the control qubit (``q_0``) indicates the state preparation.
        Since this experiment requires two sets of sub experiments with the control qubit in the
        excited and ground state, ``P`` will become ``X`` gate or just be omitted, respectively.
        Here ``cr_tone`` is implemented by a single cross resonance tone
        driving the control qubit at the frequency of the target qubit.
        The pulse envelope is the flat-topped Gaussian implemented by the parametric pulse
        :py:class:`~qiskit.pulse.library.parametric_pulses.GaussianSquare`.

        This experiment scans the flat-top width of the :py:class:`~qiskit.pulse.library.\
        parametric_pulses.GaussianSquare` envelope with the fixed rising and falling edges.
        The total pulse duration is implicitly computed to meet the timing constraints of
        the target backend. The edge duration is usually computed as

        .. math::

            \tau_{\rm edges} = 2 r \sigma,

        where the :math:`r` is the ratio of the actual edge duration to :math:`\sigma` of
        the Gaussian rising and falling edges. Note that actual edge duration is not
        identical to the net duration because of the smaller pulse amplitude of the edges.

        The net edge duration is an extra fitting parameter with initial guess

        .. math::

            \tau_{\rm edges}' = \sqrt{2 \pi} \sigma,

        which is derived by assuming a square edges with the full pulse amplitude.

    # section: analysis_ref
        :py:class:`CrossResonanceHamiltonianAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1603.04821

    # section: tutorial
        .. ref_website:: Qiskit Textbook 6.7,
            https://qiskit.org/textbook/ch-quantum-hardware/hamiltonian-tomography.html
    """
    __n_echos = 1

    # Fully parametrize CR pulse. This is because parameters can be updated at anytime
    # through experiment options, but CR schedule defined in the batch experiment
    # is immediately passed to the component experiments at the class instantiation.
    __parameters = {
        "amp": circuit.Parameter("amp"),
        "amp_t": circuit.Parameter("amp_t"),
        "sigma": circuit.Parameter("sigma"),
        "risefall": circuit.Parameter("risefall"),
        "duration": circuit.Parameter("duration"),
        "cr_channel": circuit.Parameter("cr_channel"),
    }

    def __init__(
        self,
        qubits: Tuple[int, int],
        flat_top_widths: Iterable[float],
        backend: Optional[Backend] = None,
        **kwargs,
    ):
        """Create a new experiment.

        Args:
            qubits: Two-value tuple of qubit indices on which to run tomography.
                The first index stands for the control qubit.
            flat_top_widths: The total duration of the square part of cross resonance pulse(s)
                to scan, in units of dt. The total pulse duration including Gaussian rising and
                falling edges is implicitly computed with experiment parameters ``sigma`` and
                ``risefall``.
            backend: Optional, the backend to run the experiment on.
            kwargs: Pulse parameters. See :meth:`experiment_options` for details.

        Raises:
            QiskitError: When ``qubits`` length is not 2.
        """
        if len(qubits) != 2:
            raise QiskitError(
                "Length of qubits is not 2. Please provide index for control and target qubit."
            )

        cal_def = self._default_cr_schedule(*qubits)

        pulse_gate = circuit.Gate(
            "cr_gate",
            num_qubits=2,
            params=cal_def.parameters,
        )

        cr_circuit = self._default_cr_sequence(pulse_gate)

        # Control state = 0
        cr_circuit0 = QuantumCircuit(2)
        cr_circuit0.compose(cr_circuit, inplace=True)
        cr_circuit0.add_calibration(
            gate=pulse_gate,
            qubits=qubits,
            schedule=cal_def,
            params=cal_def.parameters,
        )
        exp0 = TomographyElement(
            qubits=qubits,
            tomography_circuit=cr_circuit0,
            backend=backend,
        )
        exp0.analysis.set_options(
            result_parameters=[
                ParameterRepr("px", "cr_tomo_px0", "rad/s"),
                ParameterRepr("py", "cr_tomo_py0", "rad/s"),
                ParameterRepr("pz", "cr_tomo_pz0", "rad/s"),
            ]
        )

        # Control state = 1
        cr_circuit1 = QuantumCircuit(2)
        cr_circuit1.x(0)
        cr_circuit1.compose(cr_circuit, inplace=True)
        cr_circuit1.add_calibration(
            gate=pulse_gate,
            qubits=qubits,
            schedule=cal_def,
            params=cal_def.parameters,
        )
        exp1 = TomographyElement(
            qubits=qubits,
            tomography_circuit=cr_circuit1,
            backend=backend,
        )
        exp1.analysis.set_options(
            result_parameters=[
                ParameterRepr("px", "cr_tomo_px1", "rad/s"),
                ParameterRepr("py", "cr_tomo_py1", "rad/s"),
                ParameterRepr("pz", "cr_tomo_pz1", "rad/s"),
            ]
        )

        super().__init__(
            experiments=[exp0, exp1],
            backend=backend,
        )
        self.analysis = CrossResonanceHamiltonianAnalysis(analyses=[exp0.analysis, exp1.analysis])
        self.set_experiment_options(flat_top_widths=flat_top_widths, **kwargs)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            flat_top_widths (np.ndarray): The total duration of the square part of
                cross resonance pulse(s) to scan, in units of dt. This can start from zero and
                take positive real values representing the durations.
                Pulse edge effect is considered as an offset to the durations.
            amp (complex): Amplitude of the cross resonance tone.
            amp_t (complex): Amplitude of the cancellation or rotary drive on target qubit.
            sigma (float): Sigma of Gaussian rise and fall edges, in units of dt.
            risefall (float): Ratio of edge durations to sigma.
        """
        options = super()._default_experiment_options()
        options.flat_top_widths = None
        options.amp = 0.2
        options.amp_t = 0.0
        options.sigma = 64
        options.risefall = 2

        return options

    def set_experiment_options(self, **fields):
        """Set the experiment options.

        Args:
            fields: The fields to update the options
        """
        super().set_experiment_options(**fields)

        # Override component experiment configurations
        opt = self.experiment_options

        pulse_parameters = {
            "amp": opt.amp,
            "amp_t": opt.amp_t,
            "sigma": opt.sigma,
            "risefall": opt.risefall,
        }

        # Entire CR pulse duration (in dt)
        t_risefall = 2 * opt.sigma * opt.risefall
        cr_durations = np.asarray(opt.flat_top_widths, dtype=float) / self.__n_echos + t_risefall

        # Effective length of Gaussian rising falling edges for fit guess (in dt).
        edge_duration = np.sqrt(2 * np.pi) * opt.sigma * self.__n_echos

        for exp in self.component_experiment():
            # Copy pulse configurations to component experiments
            exp.set_experiment_options(
                durations=cr_durations,
                pulse_parameters=pulse_parameters,
                t_risefall=t_risefall,
                xval_offset=edge_duration,
            )

    def set_transpile_options(self, **fields):
        """Set the transpiler options for :meth:`run` method.

        Args:
            fields: The fields to update the options
        """
        super().set_transpile_options(fields)

        for exp in self.component_experiment():
            exp.set_transpile_options(fields)

    @classmethod
    def _default_cr_sequence(cls, pulse_gate: circuit.Gate) -> circuit.QuantumCircuit:
        """Circuit level representation of cross resonance sequence.

        Args:
            pulse_gate: Gate definition of the cross resonance.

        Returns:
            QuantumCircuit representation of cross resonance sequence.
        """
        cr_circuit = circuit.QuantumCircuit(2)
        cr_circuit.append(pulse_gate, [0, 1])

        return cr_circuit

    @classmethod
    def _default_cr_schedule(cls, control_index, target_index) -> pulse.Schedule:
        """Pulse level representation of single cross resonance gate.

        Args:
            control_index: Index of control qubit.
            target_index: Index of target qubit.

        Returns:
            Pulse schedule of cross resonance.
        """
        with pulse.build(default_alignment="left", name="cr") as cal_def:

            # add cross resonance tone
            pulse.play(
                pulse.GaussianSquare(
                    duration=cls.__parameters["duration"],
                    amp=cls.__parameters["amp"],
                    sigma=cls.__parameters["sigma"],
                    risefall_sigma_ratio=cls.__parameters["risefall"],
                ),
                pulse.ControlChannel(cls.__parameters["cr_channel"]),
            )
            pulse.play(
                pulse.GaussianSquare(
                    duration=cls.__parameters["duration"],
                    amp=cls.__parameters["amp"],
                    sigma=cls.__parameters["sigma"],
                    risefall_sigma_ratio=cls.__parameters["risefall"],
                ),
                pulse.DriveChannel(target_index),
            )

            # place holder for empty drive channels. this is necessary due to known pulse gate bug.
            pulse.delay(cls.__parameters["duration"], pulse.DriveChannel(control_index))

        return cal_def


class EchoedCrossResonanceHamiltonian(CrossResonanceHamiltonian):
    r"""Echoed cross resonance Hamiltonian tomography experiment.

    # section: overview

        This is a variant of :py:class:`CrossResonanceHamiltonian`
        for which the experiment framework is identical but the
        cross resonance operation is realized as an echoed sequence
        to remove unwanted single qubit rotations. The cross resonance
        circuit looks like:

        .. parsed-literal::

                 ┌────────────────────┐  ┌───┐  ┌────────────────────┐
            q_0: ┤0                   ├──┤ X ├──┤0                   ├──────────
                 │  cr_tone(duration) │┌─┴───┴─┐│  cr_tone(duration) │┌────────┐
            q_1: ┤1                   ├┤ Rz(π) ├┤1                   ├┤ Rz(-π) ├
                 └────────────────────┘└───────┘└────────────────────┘└────────┘

        Here two ``cr_tone``s are applied where the latter one is with the
        control qubit state flipped and with a phase flip of the target qubit frame.
        This operation is equivalent to applying the ``cr_tone`` with a negative amplitude.
        The Hamiltonian for this decomposition has no IX and ZI interactions,
        and also a reduced IY interaction to some extent (not completely eliminated) [1].
        Note that the CR Hamiltonian tomography experiment cannot detect the ZI term.
        However, it is sensitive to the IX and IY terms.

    # section: reference
        .. ref_arxiv:: 1 2007.02925

    # see_also:
        qiskit_experiments.library.characterization.CrossResonanceHamiltonian

    """

    __n_echos = 2

    @classmethod
    def _default_cr_sequence(cls, pulse_gate: circuit.Gate) -> circuit.QuantumCircuit:
        """Circuit level representation of cross resonance sequence.

        Args:
            pulse_gate: Gate definition of the cross resonance.

        Returns:
            QuantumCircuit representation of cross resonance sequence.
        """
        cr_circuit = QuantumCircuit(2)
        cr_circuit.append(pulse_gate, [0, 1])
        cr_circuit.x(0)
        cr_circuit.rz(np.pi, 1)
        cr_circuit.append(pulse_gate, [0, 1])
        cr_circuit.rz(-np.pi, 1)

        return cr_circuit
