from os import environ

import numpy as np
import pytest
from pyquil import Program
from pyquil.gates import CNOT, MEASURE, RX, H
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Declare

from pyquil_azure import AzureQuantumComputer, get_qpu, get_qvm

TEST_PROGRAM = Program(
    Declare("ro", "BIT", 2),
    H(0),
    CNOT(0, 1),
    MEASURE(0, ("ro", 0)),
    MEASURE(1, ("ro", 1)),
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


# noinspection PyUnresolvedReferences
@pytest.fixture(scope="module", params=["QVM", "QPU"])  # type: ignore
def qc(request: pytest.FixtureRequest, qpu: AzureQuantumComputer, qvm: AzureQuantumComputer) -> AzureQuantumComputer:
    if request.param == "QPU":
        return qpu
    elif request.param == "QVM":
        return qvm
    else:
        raise ValueError("Invalid params")


@pytest.fixture(scope="module")  # type: ignore
def qpu() -> AzureQuantumComputer:
    quantum_processor_id = environ.get("TEST_QUANTUM_PROCESSOR")

    if quantum_processor_id is None:
        raise Exception("'TEST_QUANTUM_PROCESSOR' env var required for e2e tests.")

    return get_qpu(
        quantum_processor_id,
    )


@pytest.fixture(scope="module")  # type: ignore
def qvm() -> AzureQuantumComputer:
    return get_qvm()


def test_basic_program(qc: AzureQuantumComputer) -> None:
    results = qc.run(qc.compile(TEST_PROGRAM)).readout_data.get("ro")

    assert results.shape == (1000, 2)


def test_parametric_program(qc: AzureQuantumComputer) -> None:
    compiled = qc.compile(
        Program(
            Declare("ro", "BIT", 1),
            Declare("theta", "REAL", 1),
            RX(MemoryReference("theta"), 0),
            MEASURE(0, ("ro", 0)),
        ).wrap_in_numshots_loop(1000),
    )

    all_results = []
    for theta in [0, np.pi, 2 * np.pi]:
        compiled.write_memory(region_name="theta", value=theta)
        results = qc.run(compiled).readout_data.get("ro")
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
    results = qpu.run(qpu.compile(QUIL_T_PROGRAM, to_native_gates=False)).readout_data.get("ro")

    assert np.mean(results) > 0.5
