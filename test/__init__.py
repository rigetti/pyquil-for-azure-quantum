"""
Contains all tests for this package.

``test_e2e_no_qcs`` is designed for integration tests that mimic how Azure users are most likely to utilize this package:
that is, running programs against Rigetti QPUs without any access to QCS.

``test_e2e_qcs_operations`` tests that the functionality of ``pyquil`` which requires QCS access (e.g., listing the
qubits for a QPU) still works through the Azure abstractions.
"""
