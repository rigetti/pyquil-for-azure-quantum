"""Test QCS operations through the AzureQuantumComputer wrappers."""

from pyquil.quil import Program

from pyquil_for_azure_quantum import AzureQuantumComputer


def test_no_error_compiling_to_native_quil(qpu: AzureQuantumComputer, basic_program: Program) -> None:
    """``quilc`` will need to pull the ISA from QCS in order to compile for a QPU. Make sure that works."""
    qpu.compiler.quil_to_native_quil(basic_program)


def test_no_error_listing_qubits(qpu: AzureQuantumComputer) -> None:
    """In order to list any part of the QPU topology, QCS credentials need to be present"""
    qpu.qubits()
