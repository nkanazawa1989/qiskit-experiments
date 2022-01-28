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

"""Tests for base experiment framework."""

from test.fake_backend import FakeBackend
from test.fake_service import FakeService
from test.fake_experiment import FakeExperiment, FakeAnalysis
from test.base import QiskitExperimentsTestCase
import ddt

from qiskit import QuantumCircuit
from qiskit_experiments.framework import ExperimentData, AnalysisResult
from qiskit_experiments.database_service.exceptions import DbExperimentDataError
from uncertainties import ufloat


@ddt.ddt
class TestFramework(QiskitExperimentsTestCase):
    """Test Base Experiment"""

    @ddt.data(None, 1, 2, 3)
    def test_job_splitting(self, max_experiments):
        """Test job splitting"""

        num_circuits = 10
        backend = FakeBackend(max_experiments=max_experiments)

        class Experiment(FakeExperiment):
            """Fake Experiment to test job splitting"""

            def circuits(self):
                """Generate fake circuits"""
                qc = QuantumCircuit(1)
                qc.measure_all()
                return num_circuits * [qc]

        exp = Experiment([0])
        expdata = exp.run(backend)
        self.assertExperimentDone(expdata)
        job_ids = expdata.job_ids

        # Comptue expected number of jobs
        if max_experiments is None:
            num_jobs = 1
        else:
            num_jobs = num_circuits // max_experiments
            if num_circuits % max_experiments:
                num_jobs += 1
        self.assertEqual(len(job_ids), num_jobs)

    def test_analysis_replace_results_true(self):
        """Test running analysis with replace_results=True"""
        analysis = FakeAnalysis()
        expdata1 = analysis.run(ExperimentData(), seed=54321)
        self.assertExperimentDone(expdata1)
        result_ids = [res.result_id for res in expdata1.analysis_results()]
        expdata2 = analysis.run(expdata1, replace_results=True, seed=12345)
        self.assertExperimentDone(expdata2)

        self.assertEqual(expdata1, expdata2)
        self.assertEqual(expdata1.analysis_results(), expdata2.analysis_results())
        self.assertEqual(result_ids, list(expdata2._deleted_analysis_results))

    def test_analysis_replace_results_false(self):
        """Test running analysis with replace_results=False"""
        analysis = FakeAnalysis()
        expdata1 = analysis.run(ExperimentData(), seed=54321)
        self.assertExperimentDone(expdata1)
        expdata2 = analysis.run(expdata1, replace_results=False, seed=12345)
        self.assertExperimentDone(expdata2)

        self.assertNotEqual(expdata1, expdata2)
        self.assertNotEqual(expdata1.experiment_id, expdata2.experiment_id)
        self.assertNotEqual(expdata1.analysis_results(), expdata2.analysis_results())

    def test_analysis_config(self):
        """Test analysis config dataclass"""
        analysis = FakeAnalysis(arg1=10, arg2=20)
        analysis.set_options(option1=False, option2=True)
        config = analysis.config()
        loaded = config.analysis()
        self.assertEqual(analysis.config(), loaded.config())
        self.assertEqual(analysis.options, loaded.options)

    def test_analysis_from_config(self):
        """Test analysis config dataclass"""
        analysis = FakeAnalysis(arg1=10, arg2=20)
        analysis.set_options(option1=False, option2=True)
        config = analysis.config()
        loaded = FakeAnalysis.from_config(config)
        self.assertEqual(config, loaded.config())

    def test_analysis_runtime_opts(self):
        """Test runtime options don't modify instance"""
        opts = {"opt1": False, "opt2": False}
        run_opts = {"opt1": True, "opt2": True, "opt3": True}
        analysis = FakeAnalysis()
        analysis.set_options(**opts)
        analysis.run(ExperimentData(), **run_opts)
        self.assertEqual(analysis.options.__dict__, opts)


class TestAnalysisResult(QiskitExperimentsTestCase):

    def test_cannot_save_result(self):
        """Analysis result cannot be saved without id or device info."""
        result = AnalysisResult(name="fake", value=123)

        # ID or device components are not populated
        with self.assertRaises(DbExperimentDataError):
            result.save()

    def test_round_trip_result(self):
        """AnalysisResult can be saved and loaded from database."""
        service = FakeService()

        result = AnalysisResult(name="test", value=123)
        result.experiment_id = "12345"
        result.device_components = ["Q1", "Q2"]
        result.service = service
        result.save()

        loaded_result = AnalysisResult.load(result_id=result.result_id, service=service)

        self.assertEqual(repr(result), repr(loaded_result))

    def test_round_trip_ufloat(self):
        """UFloat value can be stored in result and saved and loaded from database."""
        service = FakeService()

        result = AnalysisResult(name="test", value=ufloat(0.1, 0.2))
        result.experiment_id = "12345"
        result.device_components = ["Q1"]
        result.service = service
        result.save()

        loaded_result = AnalysisResult.load(result_id=result.result_id, service=service)

        self.assertEqual(repr(result), repr(loaded_result))

    def test_round_trip_ufloat_with_unit(self):
        """UFloat value with unit can be stored in result and saved and loaded from database."""
        service = FakeService()

        result = AnalysisResult(name="test", value=ufloat(0.1, 0.2), unit="Hz")
        result.experiment_id = "12345"
        result.device_components = ["Q1"]
        result.service = service
        result.save()

        loaded_result = AnalysisResult.load(result_id=result.result_id, service=service)

        self.assertEqual(repr(result), repr(loaded_result))

    def test_round_trip_ufloat_operable(self):
        """Loaded UFloat values can be operated with variance."""
        service = FakeService()

        result1 = AnalysisResult(name="test", value=ufloat(12.1, 3.0))
        result1.experiment_id = "12345"
        result1.device_components = ["Q1"]
        result1.service = service
        result1.save()

        result2 = AnalysisResult(name="test", value=ufloat(15.6, 4.0))
        result2.experiment_id = "12345"
        result2.device_components = ["Q1"]
        result2.service = service
        result2.save()

        loaded_result1 = AnalysisResult.load(result_id=result1.result_id, service=service)
        loaded_result2 = AnalysisResult.load(result_id=result2.result_id, service=service)

        new_val = loaded_result1.value + loaded_result2.value

        self.assertEqual(new_val.n, 27.7)
        self.assertEqual(new_val.s, 5.0)

    def test_expdata_ufloat_operable(self):
        """UFloat values saved in the experiment data can be operated with variance."""
        expdata = ExperimentData()

        result1 = AnalysisResult(name="test1", value=ufloat(12.1, 3.0))
        result1.experiment_id = "12345"
        result1.device_components = ["Q1"]

        result2 = AnalysisResult(name="test2", value=ufloat(15.6, 4.0))
        result2.experiment_id = "12345"
        result2.device_components = ["Q1"]

        expdata.add_analysis_results([result1, result2])

        loaded_result1 = expdata.analysis_results("test1")
        loaded_result2 = expdata.analysis_results("test2")

        new_val = loaded_result1.value + loaded_result2.value

        self.assertEqual(new_val.n, 27.7)
        self.assertEqual(new_val.s, 5.0)

    def test_round_trip_expdata_analysis(self):
        """Test round trip analysis result via experiment data."""
        service = FakeService()

        expdata = ExperimentData(backend=FakeBackend())
        expdata.service = service

        result = AnalysisResult(name="test", value="some_value")
        result.experiment_id = expdata.experiment_id
        result.device_components = ["Q1"]

        expdata.add_analysis_results(result)
        expdata.save()

        loaded_expdata = ExperimentData.load(experiment_id=expdata.experiment_id, service=service)
        loaded_result = loaded_expdata.analysis_results("test")

        self.assertEqual(repr(result), repr(loaded_result))

    def test_round_trip_composit_expdata_analysis(self):
        """Test round trip analysis result via composite experiment data."""
        service = FakeService()

        expdata1 = ExperimentData(backend=FakeBackend())
        result1 = AnalysisResult(name="test1", value="foo", quality="good", extra={"meta1": 1})
        result1.experiment_id = expdata1.experiment_id
        result1.device_components = ["Q1"]
        expdata1.add_analysis_results(result1)

        expdata2 = ExperimentData(backend=FakeBackend())
        result2 = AnalysisResult(name="test2", value="boo", quality="bad", extra={"meta2": 2})
        result2.experiment_id = expdata2.experiment_id
        result2.device_components = ["Q2"]
        expdata2.add_analysis_results(result2)

        comp_expdata = ExperimentData(backend=FakeBackend(), child_data=[expdata1, expdata2])
        comp_expdata.service = service
        comp_expdata.save()

        loaded_expdata = ExperimentData.load(experiment_id=comp_expdata.experiment_id, service=service)
        loaded_result1 = loaded_expdata.child_data(0).analysis_results("test1")
        loaded_result2 = loaded_expdata.child_data(1).analysis_results("test2")

        self.assertEqual(repr(result1), repr(loaded_result1))
        self.assertEqual(repr(result2), repr(loaded_result2))
