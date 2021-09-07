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
Composite Experiment abstract base class.
"""

from abc import abstractmethod
from typing import List, Optional
import warnings

from qiskit import QuantumCircuit
from qiskit.providers import Backend, BaseJob
from qiskit.exceptions import QiskitError
from qiskit_experiments.framework import BaseExperiment, ExperimentData
from .composite_experiment_data import CompositeExperimentData
from .composite_analysis import CompositeAnalysis


class CompositeExperiment(BaseExperiment):
    """Composite Experiment base class"""

    __analysis_class__ = CompositeAnalysis
    __experiment_data__ = CompositeExperimentData

    def __init__(self, experiments, qubits, experiment_type=None):
        """Initialize the composite experiment object.

        Args:
            experiments (List[BaseExperiment]): a list of experiment objects.
            qubits (int or Iterable[int]): the number of qubits or list of
                                           physical qubits for the experiment.
            experiment_type (str): Optional, composite experiment subclass name.
        """
        self._experiments = experiments
        self._num_experiments = len(experiments)
        super().__init__(qubits, experiment_type=experiment_type)

    def run_transpile(self, backend: Backend, **options) -> List[QuantumCircuit]:
        """Run transpile and returns transpiled circuits.

        Args:
            backend: Target backend.
            options: User provided runtime options.

        Returns:
            Transpiled circuit to execute.

        Note:
            This is transpile method for the composite experiment subclass.
            This internally calls the transpile method of the nested experiments and
            flattens the list of sub circuits generated by each experiment.
            Note that transpile is called for individual circuit, and thus transpile
            configurations and hook methods are separately applied.

            No transpile configuration assumed for composite experiment object itself.
        """
        # Transpile each sub experiment circuit
        circuits = list(map(lambda expr: expr.run_transpile(backend), self._experiments))

        # This is not identical to the `num_qubits` when the backend is AerSimulator.
        # In this case, usually a circuit qubit number is determined by the maximum qubit index.
        n_qubits = max(max(sub_circ.num_qubits for sub_circ in sub_circs) for sub_circs in circuits)

        # merge circuits
        return self._flatten_circuits(circuits, n_qubits)

    def run_analysis(
        self, experiment_data: ExperimentData, job: BaseJob = None, **options
    ) -> ExperimentData:
        """Run analysis and update ExperimentData with analysis result.

        Args:
            experiment_data: The experiment data to analyze.
            job: The future object of experiment result which is currently running on the backend.
            options: Additional analysis options. Any values set here will
                override the value from :meth:`analysis_options` for the current run.

        Returns:
            An experiment data object containing the analysis results and figures.

        Raises:
            QiskitError: When the experiment data format is not for the composite experiment.

        Note:
            This is analysis method for the composite experiment subclass.
            This internally calls the analysis method of the nested experiments and
            outputs a representative data entry for the composite analysis.
            Note that analysis is called for individual experiment data, and thus analysis
            configurations and hook methods are separately applied.

            No analysis configuration assumed for composite experiment object itself.
        """
        if not isinstance(experiment_data, CompositeExperimentData):
            raise QiskitError("CompositeAnalysis must be run on CompositeExperimentData.")

        if len(options) > 0:
            warnings.warn(
                f"Analysis options for the composite experiment are provided: {options}. "
                "Note that the provided options will override every analysis of an experiment"
                "associated with this composite experiment.",
                UserWarning,
            )

        return super().run_analysis(experiment_data, job, **options)

    @abstractmethod
    def _flatten_circuits(
        self,
        circuits: List[List[QuantumCircuit]],
        num_qubits: int,
    ) -> List[QuantumCircuit]:
        """An abstract method to control merger logic of sub experiments."""
        pass

    def circuits(self, backend: Optional[Backend] = None):
        """Composite experiment does not provide this method.

        Args:
            backend: The targe backend.

        Raises:
            QiskitError: When this method is called.
        """
        raise QiskitError(
            f"{self.__class__.__name__} does not generate experimental circuits by itself. "
            "Call the corresponding method of individual experiment class to find circuits, "
            "or call `run_transpile` method to get circuits run on the target backend."
        )

    @property
    def num_experiments(self):
        """Return the number of sub experiments"""
        return self._num_experiments

    def component_experiment(self, index=None):
        """Return the component Experiment object.
        Args:
            index (int): Experiment index, or ``None`` if all experiments are to be returned.
        Returns:
            BaseExperiment: The component experiment(s).
        """
        if index is None:
            return self._experiments
        return self._experiments[index]

    def component_analysis(self, index):
        """Return the component experiment Analysis object"""
        return self.component_experiment(index).analysis()

    def _add_job_metadata(self, experiment_data, job, **run_options):
        # Add composite metadata
        super()._add_job_metadata(experiment_data, job, **run_options)

        # Add sub-experiment options
        for i in range(self.num_experiments):
            sub_exp = self.component_experiment(i)

            # Run and transpile options are always overridden
            if (
                sub_exp.run_options != sub_exp._default_run_options()
                or sub_exp.transpile_options != sub_exp._default_transpile_options()
            ):

                warnings.warn(
                    "Sub-experiment run and transpile options"
                    " are overridden by composite experiment options."
                )
            sub_data = experiment_data.component_experiment_data(i)
            sub_exp._add_job_metadata(sub_data, job, **run_options)
