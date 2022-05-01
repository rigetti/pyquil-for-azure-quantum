"""Contains shared fixtures for testing"""
from os import environ

import pytest
from pyquil.gates import CNOT, MEASURE, H
from pyquil.quil import Program
from pyquil.quilbase import Declare

from pyquil_azure_quantum import AzureQuantumComputer, get_qpu, get_qvm


@pytest.fixture(scope="module", params=["QVM", "QPU"])
# pylint: disable-next=redefined-outer-name,invalid-name
def qc(request: pytest.FixtureRequest, qpu: AzureQuantumComputer, qvm: AzureQuantumComputer) -> AzureQuantumComputer:
    """Parametrized fixture for running on both QVM and the selected QPU."""
    if request.param == "QPU":  # type: ignore
        return qpu
    if request.param == "QVM":  # type: ignore
        return qvm
    raise ValueError("Invalid params")


@pytest.fixture(scope="module")
def qpu() -> AzureQuantumComputer:
    """A fixture for running programs on a QPU as defined in a TEST_QUANTUM_PROCESSOR environment variable."""
    quantum_processor_id = environ.get("TEST_QUANTUM_PROCESSOR")

    if quantum_processor_id is None:
        raise Exception("'TEST_QUANTUM_PROCESSOR' env var required for e2e tests.")

    return get_qpu(
        quantum_processor_id,
    )


@pytest.fixture(scope="module")
def qvm() -> AzureQuantumComputer:
    """A fixture for running programs on QVM through Azure Quantum"""
    return get_qvm()


@pytest.fixture(scope="module")
def basic_program() -> Program:
    """A fixture which returns a basic bell state ``pyquil.quil.Program``"""

    return Program(
        Declare("ro", "BIT", 2),
        H(0),
        CNOT(0, 1),
        MEASURE(0, ("ro", 0)),
        MEASURE(1, ("ro", 1)),
    ).wrap_in_numshots_loop(1000)
