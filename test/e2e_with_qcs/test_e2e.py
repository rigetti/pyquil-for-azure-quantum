from os import environ

import pytest
from pyquil import Program
from pyquil.gates import CNOT, MEASURE, H
from pyquil.quilbase import Declare

from pyquil_azure import AzureQuantumComputer, get_qpu

TEST_PROGRAM = Program(
    Declare("ro", "BIT", 2),
    H(0),
    CNOT(0, 1),
    MEASURE(0, ("ro", 0)),
    MEASURE(1, ("ro", 1)),
).wrap_in_numshots_loop(1000)


@pytest.fixture(scope="module")  # type: ignore
def qpu() -> AzureQuantumComputer:
    quantum_processor_id = environ.get("TEST_QUANTUM_PROCESSOR")

    if quantum_processor_id is None:
        raise Exception("'TEST_QUANTUM_PROCESSOR' env var required for e2e tests.")

    return get_qpu(
        quantum_processor_id,
    )


def test_no_error_compiling_to_native_quil(qpu: AzureQuantumComputer) -> None:
    qpu.compiler.quil_to_native_quil(TEST_PROGRAM)


def test_no_error_listing_qubits(qpu: AzureQuantumComputer) -> None:
    qpu.qubits()
