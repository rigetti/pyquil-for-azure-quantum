#    Copyright 2022 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""Run pyquil programs through Azure Quantum

   Usage:
      * Instead of using ``pyquil.get_qc()`` use ``pyquil_for_azure_quantum.get_qpu()`` for QPUs and
        ``pyquil_for_azure_quantum.get_qvm()`` for QVM.
      * QVM and ``quilc`` do not need to be installed or running locally.
"""

__all__ = ["get_qpu", "get_qvm", "AzureQuantumComputer", "AzureProgram"]

from dataclasses import dataclass
from os import environ
from typing import Dict, List, Optional, Tuple, Union, cast

from azure.quantum import Job, Workspace
from azure.quantum.target.rigetti import InputParams, Result, Rigetti
from lazy_object_proxy import Proxy
from numpy import array, split
from pyquil.api import QAM, QAMExecutionResult, QuantumComputer, get_qc
from pyquil.quil import Program
from wrapt import ObjectProxy

ParamValue = Union[int, float]


# noinspection PyAbstractClass
# pylint: disable-next=abstract-method
class AzureProgram(ObjectProxy, Program):  # type: ignore
    """A wrapper around a ``Program`` that is used to execute on Azure Quantum."""

    def __init__(
        self,
        program: Program,
        skip_quilc: bool,
        memory_map: Optional[Dict[str, List[List[ParamValue]]]] = None,
    ) -> None:
        super().__init__(program)
        self.skip_quilc = skip_quilc
        self.memory_map = memory_map

    def copy(self) -> "AzureProgram":
        """Perform a shallow copy of this program.

        QuilAtom and AbstractInstruction objects should be treated as immutable to avoid strange behavior when
        performing a copy.
        """
        return AzureProgram(self.__wrapped__.copy(), self.skip_quilc)

    def get_memory(self) -> Optional[Dict[str, List[List[ParamValue]]]]:
        """Retrieve the memory map for this program formatted in the way that Azure expects"""
        if self.memory_map is not None:
            return self.memory_map
        memory = self._memory.values
        if len(memory) == 0:
            return None
        memory_indexes_and_values_per_name: Dict[str, List[Tuple[int, ParamValue]]] = {}
        for ref, value in memory.items():
            if ref.name not in memory_indexes_and_values_per_name:
                memory_indexes_and_values_per_name[ref.name] = []
            memory_indexes_and_values_per_name[ref.name].append((ref.index, value))
        memory_indexes_and_values_per_name = {
            k: sorted(v, key=lambda x: x[0]) for k, v in memory_indexes_and_values_per_name.items()
        }
        substitutions: Dict[str, List[List[float]]] = {}
        for name, indexes_and_values in memory_indexes_and_values_per_name.items():
            values = []
            expected_index = 0
            for index, value in indexes_and_values:
                while index != expected_index:
                    # Pad missing values with zeros, just like pyquil does when patching
                    values.append(0.0)
                    expected_index += 1
                values.append(value)
                expected_index += 1
            substitutions[name] = [values]
        return substitutions


# pylint: disable-next=too-few-public-methods
class AzureQuantumComputer(QuantumComputer):
    """
    A ``pyquil.QuantumComputer`` that runs on Azure Quantum.
    """

    def __init__(self, *, target: str, qpu_name: str):
        qam = AzureQuantumMachine(target=target)
        compiler = Proxy(lambda: get_qc(qpu_name).compiler)
        super().__init__(name=qpu_name, qam=qam, compiler=compiler)

    # pylint: disable-next=no-self-use,unused-argument
    def compile(
        self,
        program: Program,
        to_native_gates: bool = True,
        optimize: bool = True,
        *,
        protoquil: Optional[bool] = None,
    ) -> AzureProgram:
        """Compile a program for Azure execution.

        By default, this stage is a no-op and is here for better compatibility with pyquil programs. Azure will do all
        compilation in the cloud. However, if you want to tell Azure to skip the quilc step, pass ``False`` to
        ``to_native_gates``. This is necessary when utilizing Quil-T. If you want to run ``quilc`` locally in order to
        check the Native Quil which will be executed, call ``compiler.quil_to_native_quil`` and pass the result to this
        function along with ``to_native_gates=False``.

        Args:
            program: The program to compile.
            to_native_gates: Whether to convert the program to native gates by running ``quilc`` *in the cloud* prior to
                execution. Defaults to ``True``, pass ``False`` to skip cloud-side ``quilc``.
            optimize: Has no effect on Azure.
            protoquil: Has no effect on Azure.

        Returns:
            An ``AzureExecutable`` which is effectively a simple wrapper around a ``Program``. No actual compilation is
            done here.
        """
        return AzureProgram(program, skip_quilc=not to_native_gates)

    def run_batch(self, executable: AzureProgram, memory_map: Dict[str, List[List[float]]]) -> List[QAMExecutionResult]:
        """Run a sequence of memory values through the program.

        See Also:
            * [`AzureQuantumMachine.run_batch`][pyquil_for_azure_quantum.AzureQuantumMachine.run_batch]
        """
        qam = cast(AzureQuantumMachine, self.qam)
        return qam.run_batch(executable, memory_map)


def get_qpu(qpu_name: str) -> AzureQuantumComputer:
    """Get an AzureQuantumComputer targeting a real QPU

    These Azure Quantum configuration environment variables __must__ be set:

    - AZURE_QUANTUM_SUBSCRIPTION_ID: The Azure subscription ID where the Quantum Workspace is located.
    - AZURE_QUANTUM_WORKSPACE_RG: The Azure resource group where the Quantum Workspace is located.
    - AZURE_QUANTUM_WORKSPACE_NAME: The name of the Quantum Workspace.
    - AZURE_QUANTUM_WORKSPACE_LOCATION: The region where the Quantum Workspace is located.

    If these are not set, this will attempt to open a browser to authenticate:

    - AZURE_CLIENT_ID
    - AZURE_CLIENT_SECRET
    - AZURE_TENANT_ID

    Raises:
        KeyError: If required environment variables are not set.
    """
    return AzureQuantumComputer(target=f"rigetti.qpu.{qpu_name.lower()}", qpu_name=qpu_name)


def get_qvm() -> AzureQuantumComputer:
    """Get an AzureQuantumComputer targeting a cloud-hosted QVM

    These Azure Quantum configuration environment variables __must__ be set:

    - AZURE_QUANTUM_SUBSCRIPTION_ID: The Azure subscription ID where the Quantum Workspace is located.
    - AZURE_QUANTUM_WORKSPACE_RG: The Azure resource group where the Quantum Workspace is located.
    - AZURE_QUANTUM_WORKSPACE_NAME: The name of the Quantum Workspace.
    - AZURE_QUANTUM_WORKSPACE_LOCATION: The region where the Quantum Workspace is located.

    If these are not set, this will attempt to open a browser to authenticate:

    - AZURE_CLIENT_ID
    - AZURE_CLIENT_SECRET
    - AZURE_TENANT_ID

    Raises:
        KeyError: If required environment variables are not set.
    """
    return AzureQuantumComputer(target="rigetti.sim.qvm", qpu_name="qvm")


@dataclass
class AzureJob:
    """Keeps track of an ``AzureProgram`` that was submitted to Azure Quantum."""

    job: Job
    executable: AzureProgram


class AzureQuantumMachine(QAM[AzureJob]):
    """An implementation of QAM which runs programs using Azure Quantum

    These Azure Quantum configuration environment variables __must__ be set:
        * AZURE_QUANTUM_SUBSCRIPTION_ID: The Azure subscription ID where the Quantum Workspace is located.
        * AZURE_QUANTUM_WORKSPACE_RG: The Azure resource group where the Quantum Workspace is located.
        * AZURE_QUANTUM_WORKSPACE_NAME: The name of the Quantum Workspace.
        * AZURE_QUANTUM_WORKSPACE_LOCATION: The region where the Quantum Workspace is located.

    Credentials for communicating with Azure Quantum should be stored in the following environment variables. If they
    are not set, this will attempt to open a browser to authenticate:
        * AZURE_CLIENT_ID
        * AZURE_CLIENT_SECRET
        * AZURE_TENANT_ID

    Raises:
        KeyError: If required environment variables are not set.
    """

    def __init__(self, *, target: str) -> None:
        self._workspace = Workspace(
            subscription_id=environ["AZURE_QUANTUM_SUBSCRIPTION_ID"],
            resource_group=environ["AZURE_QUANTUM_WORKSPACE_RG"],
            name=environ["AZURE_QUANTUM_WORKSPACE_NAME"],
            location=environ["AZURE_QUANTUM_WORKSPACE_LOCATION"],
        )
        # noinspection PyTypeChecker
        self._target = Rigetti(
            workspace=self._workspace,
            name=target,
        )

    # pylint: disable-next=useless-super-delegation
    def run(self, executable: AzureProgram) -> QAMExecutionResult:  # type: ignore[override]
        """Run the executable and wait for its results"""
        return super().run(executable)

    def execute(self, executable: AzureProgram, name: str = "pyquil-azure-job") -> AzureJob:  # type: ignore[override]
        """Run an AzureProgram on Azure Quantum. Unlike normal QAM this does not accept a ``QuantumExecutable``.

        You should build the ``AzureProgram`` via ``AzureQuantumComputer.compile``.

        Args:
            executable: The AzureProgram to run.
            name: An optional name for the job which will show up in the Azure Quantum UI.

        Returns:
            A ``Job`` which can be used to check the status of the job or retrieve a ``QAMExecutionResult`` via
            ``get_result()``.
        """
        executable = executable.copy()
        input_params = InputParams(
            count=executable.num_shots,
            skip_quilc=executable.skip_quilc,
        )
        memory = executable.get_memory()
        if memory is not None:
            input_params.substitutions = memory

        job = self._target.submit(
            str(executable),
            name=name,
            input_params=input_params,
        )
        return AzureJob(job=job, executable=executable)

    # pylint: disable-next=no-self-use
    def get_result(self, execute_response: AzureJob) -> QAMExecutionResult:
        """Wait for a ``Job`` to complete, then return the results

        Raises:
            RuntimeError: If the job fails.
        """
        job = execute_response.job
        job.wait_until_completed()
        result = Result(job)
        numpified = {k: array(v) for k, v in result.data_per_register.items()}
        return QAMExecutionResult(
            executable=execute_response.executable,
            readout_data=numpified,
        )

    def run_batch(
        self, executable: AzureProgram, memory_map: Dict[str, List[List[float]]], name: str = "pyquil-azure-job"
    ) -> List[QAMExecutionResult]:
        """Run the executable for each set of parameters in the ``memory_map``.

        Args:
            executable: The AzureProgram to run.
            memory_map: A dictionary mapping parameter names to lists of parameter values. Each value is a list as long
                as the number of slots in the register. So if the register was ``DECLARE theta REAL[2]`` then the key
                in the dictionary would be ``theta`` and the value would be a list of lists of length 2. The entire
                program will be run (for shot count) as many times as there are values in the list. **All values (outer
                lists) must be of the same length**.
            name: An optional name for the job which will show up in the Azure Quantum UI.

        Returns:
            A list of ``QAMExecutionResult`` objects, one for each set of parameters.

        ```pycon

        >>> import numpy as np
        >>> from pyquil import Program
        >>> from pyquil.gates import CNOT, MEASURE, RX, H
        >>> from pyquil.quilatom import MemoryReference
        >>> from pyquil.quilbase import Declare
        >>> from pyquil_for_azure_quantum import get_qvm
        >>> qvm = get_qvm()
        >>> program = Program(\
             Declare("ro", "BIT", 1), \
             Declare("theta", "REAL", 1), \
             RX(MemoryReference("theta"), 0), \
             MEASURE(0, ("ro", 0)), \
        ).wrap_in_numshots_loop(1000)
        >>> compiled = qvm.compile(program)
        >>> results = qvm.run_batch(compiled, {"theta": [[0.0], [np.pi], [2 * np.pi]]})
        >>> assert len(results) == 3  # 3 values for theta—each a list of length 1
        >>> results_0 = results[0].readout_data["ro"]
        >>> assert len(results_0) == 1000  # 1000 shots
        >>> assert np.mean(results_0) == 0
        >>> results_pi = results[1].readout_data["ro"]
        >>> assert len(results_pi) == 1000
        >>> assert np.mean(results_pi) == 1

        ```
        """
        num_params = None
        for param_name, param_values in memory_map.items():
            if num_params is None:
                num_params = len(param_values)
            elif num_params != len(param_values):
                raise ValueError(
                    "All parameter values must be of the same length. "
                    f"{param_name} has length {len(param_values)} but {num_params} were expected."
                )

        input_params = InputParams(
            count=executable.num_shots, skip_quilc=executable.skip_quilc, substitutions=memory_map
        )

        job = self._target.submit(
            str(executable),
            name=name,
            input_params=input_params,
        )
        combined_result = self.get_result(AzureJob(job, executable))
        if num_params is None or num_params == 1:
            return [combined_result]
        split_results = split(combined_result.readout_data["ro"], num_params)  # type: ignore
        return [QAMExecutionResult(executable, {"ro": result}) for result in split_results]


_RawData = Union[int, float, List[float]]
