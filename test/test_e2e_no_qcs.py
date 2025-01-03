"""Test running programs on Azure Quantum without any QCS credentials"""

from typing import Dict, List, cast

import numpy as np
from pyquil.api import MemoryMap
from pyquil.gates import MEASURE, RX
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Declare

from pyquil_for_azure_quantum import AzureQuantumComputer, AzureQuantumMachine, make_substitutions_from_memory_maps

PARAMETRIZED = Program(
    Declare("ro", "BIT", 1),
    Declare("theta", "REAL", 1),
    RX(MemoryReference("theta"), 0),
    MEASURE(0, ("ro", 0)),
).wrap_in_numshots_loop(1000)
QUIL_T_PROGRAM = Program(
    """
PRAGMA INITIAL_REWIRING "PARTIAL"
DECLARE ro BIT
RX(pi) 0
FENCE 0
DELAY 0 "rf" 1e-6
MEASURE 0 ro
"""
).wrap_in_numshots_loop(10)


# pylint: disable-next=invalid-name
def test_basic_program(qc: AzureQuantumComputer, basic_program: Program) -> None:
    """A smoke test of a very basic program running through Azure Quantum with minimal assertions"""
    results = qc.run(qc.compile(basic_program)).get_register_map().get("ro")
    assert results is not None
    assert results.shape == (1000, 2)


# pylint: disable-next=invalid-name
def test_parametric_program(qc: AzureQuantumComputer) -> None:
    """Ensure that the standard ``write_memory`` -> ``run`` workflow is functional."""
    compiled = qc.compile(PARAMETRIZED)

    all_results = []
    for theta in [0, np.pi, 2 * np.pi]:
        memory_map: Dict[str, List[float]] = {"theta": [theta]}
        results = qc.run(executable=compiled, memory_map=memory_map).get_register_map().get("ro")
        assert results is not None
        all_results.append(np.mean(results))

    if qc.name != "qvm":
        assert all_results[0] < 0.2
        assert all_results[1] > 0.8
        assert all_results[2] < 0.2
    else:
        assert all_results[0] == 0.0
        assert all_results[1] > 0.8
        assert all_results[2] == 0.0


def test_quil_t(qpu: AzureQuantumComputer) -> None:
    """Test skipping ``quilc`` on the backend by passing a program which will not compile (Quil-T)"""
    results = qpu.run(qpu.compile(QUIL_T_PROGRAM, to_native_gates=False)).get_register_map().get("ro")
    assert results is not None

    assert np.mean(results) > 0.5


def test_memory_maps_to_substitutions() -> None:
    executions = [{"theta": [value]} for value in [0, np.pi, 2 * np.pi]]
    substitutions = make_substitutions_from_memory_maps(executions)
    assert substitutions is not None
    assert substitutions.keys() == {"theta"}
    assert len(substitutions["theta"]) == 3


# pylint: disable-next=invalid-name
def test_run_batch(qc: AzureQuantumComputer) -> None:
    """Test the ``run_batch`` interface which should be much faster than normal parametrization"""
    compiled = qc.compile(PARAMETRIZED)

    executions = [{"theta": [value]} for value in [0, np.pi, 2 * np.pi]]
    results = qc.run_batch(compiled, executions)

    results_0 = results[0].get_register_map().get("ro")
    assert results_0 is not None
    results_0_mean = np.mean(results_0)
    results_pi = results[1].get_register_map().get("ro")
    assert results_pi is not None
    results_pi_mean = np.mean(results_pi)
    results_2pi = results[2].get_register_map().get("ro")
    assert results_2pi is not None
    results_2pi_mean = np.mean(results_2pi)
    if qc.name == "qvm":
        assert results_0_mean == 0.0
        assert results_pi_mean == 1.0
        assert results_2pi_mean == 0.0
    else:
        assert results_0_mean < 0.2
        assert results_pi_mean > 0.8
        assert results_2pi_mean < 0.2
