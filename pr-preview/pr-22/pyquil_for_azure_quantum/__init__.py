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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union, cast

from azure.quantum import Job, Workspace
from azure.quantum.target.rigetti import InputParams, Result, Rigetti, RigettiTarget
from lazy_object_proxy import Proxy
from numpy import split
from pyquil.api import QAM, MemoryMap, QAMExecutionResult, QuantumComputer, get_qc
from pyquil.quil import Program
from qcs_sdk import ExecutionData, RegisterData, ResultData  # pylint: disable=no-name-in-module
from qcs_sdk.qvm import QVMResultData  # pylint: disable=no-name-in-module
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
    ) -> None:
        super().__init__(program)
        self.skip_quilc = skip_quilc

    def copy(self) -> "AzureProgram":
        """Perform a shallow copy of this program.

        QuilAtom and AbstractInstruction objects should be treated as immutable to avoid strange behavior when
        performing a copy.
        """
        return AzureProgram(self.__wrapped__.copy(), self.skip_quilc)


# pylint: disable-next=too-few-public-methods
# AzureQuantumComputer provides an API for executing jobs using the Rigetti
# Azure integration. We intentionally do not support running arbitrary QCS
# `Program` objects with this class; `AzureProgram` is used instead. Our
# overrides are thus incompatible with the base class, so we disable the type
# check wherever `AzureProgram` is used as an argument.
class AzureQuantumComputer(QuantumComputer):
    """
    A ``pyquil.QuantumComputer`` that runs on Azure Quantum.
    """

    def __init__(self, *, target: str, qpu_name: str):
        # pylint: disable=abstract-class-instantiated
        qam = AzureQuantumMachine(target=target)
        compiler = Proxy(lambda: get_qc(qpu_name).compiler)
        super().__init__(name=qpu_name, qam=qam, compiler=compiler)

    # pylint: disable-next=unused-argument
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

    def run_with_memory_map_batch(
        self,
        # See the comment on the `AzureQuantumComputer` class for why the type error is ignored.
        executable: AzureProgram,  # type: ignore[override]
        memory_maps: Iterable[MemoryMap],
        **__kwargs: Any,
    ) -> List[QAMExecutionResult]:
        """Run the executable for each of the ``memory_maps``.

        Args:
            executable: The AzureProgram to run.
            memory_maps: An iterable containing ``MemoryMaps`` with desired mappings of parameter names to parameter values.
                Each value is a list as long as the number of elements in the register. So if the register was ``DECLARE theta REAL[2]``
                then the key in the dictionary would be ``theta`` and the value would be a list of length 2. The entire program
                will be run (for shot count) once for each ``MemoryMap``. (If no memory maps are provided, the program will be run once.)
            name: An optional name for the job which will show up in the Azure Quantum UI.

        Returns:
            A list of ``QAMExecutionResult`` objects, one for each ``MemoryMap``.

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
        >>> results = qvm.run_with_memory_map_batch(compiled, [{"theta": [value]} for value in [0.0, np.pi, 2 * np.pi]}])
        >>> assert len(results) == 3  # 3 values for theta—each a list of length 1
        >>> results_0 = results[0].readout_data["ro"]
        >>> assert len(results_0) == 1000  # 1000 shots
        >>> assert np.mean(results_0) == 0
        >>> results_pi = results[1].readout_data["ro"]
        >>> assert len(results_pi) == 1000
        >>> assert np.mean(results_pi) == 1

        ```

        See Also:
            * [`AzureQuantumMachine.execute_with_memory_map_batch`][pyquil_for_azure_quantum.AzureQuantumMachine.execute_with_memory_map_batch]
        """
        qam = cast(AzureQuantumMachine, self.qam)
        azure_job = qam.execute_with_memory_map_batch(executable, memory_maps)
        # `execute_with_memory_map_batch` always returns a list of length 1.
        assert len(azure_job) == 1

        combined_result = qam.get_result(azure_job[0])

        # We expect that `memory_maps` is always a list already, so this should
        # not have a performance impact.
        num_executions = len(list(memory_maps))
        if num_executions in (0, 1):
            return [combined_result]

        ro_matrix = combined_result.data.result_data.to_register_map().get_register_matrix("ro")
        if ro_matrix is None:
            return []

        return [
            QAMExecutionResult(
                executable,
                ExecutionData(
                    ResultData.from_qvm(
                        QVMResultData.from_memory_map(memory={"ro": RegisterData(split_result.tolist())})
                    )
                ),
            )
            for split_result in split(ro_matrix.to_ndarray(), num_executions)
        ]


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
    return AzureQuantumComputer(target=RigettiTarget.QVM.value, qpu_name="qvm")


@dataclass
class AzureJob:
    """Keeps track of an ``AzureProgram`` that was submitted to Azure Quantum."""

    job: Job
    executable: AzureProgram


# AzureQuantumMachine, like AzureQuantumComputer, intentionally does not support
# running arbitrary QCS `Program` objects; `AzureProgram` is used instead. Our
# overrides are thus incompatible with the base class, so we disable the type
# check wherever `AzureProgram` is used as an argument.
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

    def execute(
        self,
        # See the comment on the `AzureQuantumMachine` class for why the type error is ignored.
        executable: AzureProgram,  # type: ignore[override]
        memory_map: Optional[MemoryMap] = None,
        **kwargs: Any,  # used for `name`, to ensure signature compatibility with QAM
    ) -> AzureJob:
        """Run an AzureProgram on Azure Quantum. Unlike normal QAM this does not accept a ``QuantumExecutable``.

        You should build the ``AzureProgram`` via ``AzureQuantumComputer.compile``.

        Args:
            executable: The AzureProgram to run.
            memory_map: An optional set of parameter names to parameter values to use for execution.
            name: An optional name for the job which will show up in the Azure Quantum UI.

        Returns:
            A ``Job`` which can be used to check the status of the job or retrieve a ``QAMExecutionResult`` via
            ``get_result()``.
        """
        name = kwargs.get(
            "name",
            "pyquil-azure-job",
        )
        executable = executable.copy()
        input_params = InputParams(
            count=executable.num_shots,
            skip_quilc=executable.skip_quilc,
            substitutions=_make_substitutions_from_memory_maps([memory_map]) if memory_map is not None else None,
        )
        job = self._target.submit(
            str(executable),
            name=name,
            input_params=input_params,
        )
        return AzureJob(job=job, executable=executable)

    def get_result(self, execute_response: AzureJob) -> QAMExecutionResult:
        """Wait for a ``Job`` to complete, then return the results

        Raises:
            RuntimeError: If the job fails.
        """
        job = execute_response.job
        job.wait_until_completed()
        try:
            result = Result(job)
        except RuntimeError as e:
            # Most Azure Quantum errors do not include the job ID, so we add it
            # here for clarity.
            raise RuntimeError(f"Azure job {job.details.id} failed: {e}") from e

        # pylint: disable-next=fixme
        # TODO: as of https://github.com/microsoft/qdk-python/blob/4d6f7f75c8c7d8467f87936b1aaef449de1e0bf6/azure-quantum/azure/quantum/target/rigetti/result.py#L47
        # both QVM and QC result shapes take the memory-map form as in the QVMResultData.
        # When the Rigetti target returns results with mappings, the QPUResultData can be constructed.
        memory = {k: RegisterData(v) for k, v in result.data_per_register.items()}
        result_data = ResultData.from_qvm(QVMResultData.from_memory_map(memory=memory))

        data = ExecutionData(result_data=result_data)
        return QAMExecutionResult(
            executable=execute_response.executable,
            data=data,
        )

    def execute_with_memory_map_batch(
        self,
        # See the comment on the `AzureQuantumMachine` class for why the type error is ignored.
        executable: AzureProgram,  # type: ignore[override]
        memory_maps: Iterable[MemoryMap],
        **kwargs: Any,  # used for `name`, to ensure signature compatibility with QAM
    ) -> List[AzureJob]:
        """Run the executable for each of the ``memory_maps``.

        Args:
            executable: The AzureProgram to run.
            memory_maps: An iterable containing ``MemoryMaps`` with desired mappings of parameter names to parameter values.
                Each value is a list as long as the number of elements in the register. So if the register was ``DECLARE theta REAL[2]``
                then the key in the dictionary would be ``theta`` and the value would be a list of length 2. The entire program
                will be run (for shot count) once for each ``MemoryMap``. (If no memory maps are provided, the program will be run once.)
            name: An optional name for the job which will show up in the Azure Quantum UI.

        Returns:
            A list of ``AzureJob`` object; because ``AzureJob`` is not
            considered done until the entire batch is done, this is always a
            list of length one.

        """
        name = kwargs.get(
            "name",
            "pyquil-azure-job",
        )
        executable = executable.copy()
        memory_maps = list(memory_maps)
        input_params = InputParams(
            count=executable.num_shots,
            skip_quilc=executable.skip_quilc,
            substitutions=_make_substitutions_from_memory_maps(memory_maps),
        )
        job = self._target.submit(
            str(executable),
            name=name,
            input_params=input_params,
        )
        azure_job = AzureJob(job=job, executable=executable)
        return [azure_job]


def _make_substitutions_from_memory_maps(
    memory_maps: Sequence[MemoryMap],
) -> Optional[Dict[str, List[List[float]]]]:
    """
    Helper function to convert a list of MemoryMaps to the format expected by Azure Quantum.
    """

    if not memory_maps:
        return None

    # Function to convert the values for a single key in a single MemoryMap to
    # the format expected by Azure Quantum
    def execution_values(value: Union[Sequence[int], Sequence[float]]) -> List[float]:
        return list(map(float, value))

    substitutions = {k: [execution_values(v)] for k, v in memory_maps[0].items()}
    for index, memory_map in enumerate(memory_maps[1:]):
        for key, memory_values in memory_map.items():
            if key not in substitutions:
                raise ValueError(
                    "All MemoryMaps must contain the same keys. "
                    f"MemoryMap {index+1} contains key {key} which was not in the first MemoryMap."
                )
            substitutions[key].append(execution_values(memory_values))
    return substitutions
