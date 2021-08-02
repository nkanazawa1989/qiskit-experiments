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
Standard RB data generation.
"""
import argparse
import json
import os

from qiskit.providers.aer import QasmSimulator

from qiskit_experiments.library import StandardRB

try:
    from .utils import create_depolarizing_noise_model, analysis_save
except ImportError:
    # we cannot use relative import from the top level module.
    from utils import create_depolarizing_noise_model, analysis_save


def _standard_rb_exp_data_gen(dir_name: str, num_samples: int, seed_val: int):
    """
    Encapsulation for different experiments attributes which in turn execute.
    The data and analysis is saved to json file via "_generate_rb_fitter_data" function.
    Args:
        dir_name(str): The directory which the program will save the data and anaysis.
        num_samples (int): Number of seeds.
        seed_val (int): Seed value.
    """
    rb_exp_name = ["rb_standard_1qubit", "rb_standard_2qubits"]
    experiments_attributes = [
        {
            "num_qubits": 1,
            "physical_qubits": [0],
            "lengths": list(range(1, 1000, 100)),
            "num_samples": num_samples,
            "seed": seed_val,
        },
        {
            "num_qubits": 2,
            "physical_qubits": [0, 1],
            "lengths": list(range(1, 200, 20)),
            "num_samples": num_samples,
            "seed": seed_val,
        },
    ]
    for idx, experiment_attributes in enumerate(experiments_attributes):
        print("Generating experiment #{}: {}".format(idx, experiment_attributes))
        _generate_rb_fitter_data(dir_name, rb_exp_name[idx], experiment_attributes)


def _generate_rb_fitter_data(dir_name: str, rb_exp_name: str, exp_attributes: dict):
    """
    Executing standard RB experiment and storing its data in json format.
    The json is composed of a list that the first element is a dictionary containing
    the experiment attributes and the second element is a list with all the experiment
    data.
    Args:
        dir_name: The json file name that the program write the data to.
        rb_exp_name: The experiment name for naming the output files.
        exp_attributes: attributes to config the RB experiment.
    """
    gate_error_ratio = {
        ((0,), "id"): 1,
        ((0,), "rz"): 0,
        ((0,), "sx"): 1,
        ((0,), "x"): 1,
        ((0, 1), "cx"): 1,
    }
    transpiled_base_gate = ["cx", "sx", "x"]
    results_file_path = os.path.join(dir_name, str(rb_exp_name + "_output_data.json"))
    analysis_file_path = os.path.join(dir_name, str(rb_exp_name + "_output_analysis.json"))
    noise_model = create_depolarizing_noise_model()
    backend = QasmSimulator()
    print("Generating RB experiment")
    rb_exp = StandardRB(
        exp_attributes["physical_qubits"],
        exp_attributes["lengths"],
        num_samples=exp_attributes["num_samples"],
        seed=exp_attributes["seed"],
    )
    rb_exp.set_analysis_options(gate_error_ratio=gate_error_ratio)
    print("Running experiment")
    experiment_obj = rb_exp.run(
        backend, noise_model=noise_model, basis_gates=transpiled_base_gate
    ).block_for_results()
    print("Done running experiment")
    experiment_obj.block_for_results()
    exp_results = experiment_obj.data()
    with open(results_file_path, "w") as json_file:
        joined_list_data = [exp_attributes, exp_results]
        json_file.write(json.dumps(joined_list_data))
    analysis_save(experiment_obj.analysis_results(), analysis_file_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Standard RB ref data generation.")
    parser.add_argument(
        "--folder",
        required=False,
        default="test/randomized_benchmarking/refdata",
        type=str,
    )
    parser.add_argument(
        "--samples",
        required=False,
        default=3,
        type=int,
    )
    parser.add_argument(
        "--seed",
        required=False,
        default=100,
        type=int,
    )
    args = parser.parse_args()

    _standard_rb_exp_data_gen(args.folder, args.samples, args.seed)
