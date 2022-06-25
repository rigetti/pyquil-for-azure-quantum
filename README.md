# pyquil-for-azure-quantum

This library allows you to use [pyQuil] to run programs on [Azure Quantum](https://azure.microsoft.com/en-us/services/quantum/) against Rigetti targets. Internally, it leverages the [azure-quantum] package.

## Usage

Generally, you use [pyQuil] normally, with a few differences:

1. Instead of `pyquil.get_qc()`, you will use either `pyquil_azure_quantum.get_qvm()` or `pyquil_azure_quantum.get_qpu()`.
2. You do not need to have `qvm` or `quilc` running in order to run programs through `pyquil_azure_quantum`. You may still run them if you wish to run QVM locally instead of passing through Azure or if you wish to precompile your programs (e.g., to inspect the exact Quil that will run).
3. You do not need a QCS account or credentials unless you wish to manually inspect the details of the QPU (e.g., list all qubits).
4. You **must** have these environment variables set:
   1. `AZURE_QUANTUM_SUBSCRIPTION_ID`: The Azure subscription ID where the Quantum Workspace is located.
   2. `AZURE_QUANTUM_WORKSPACE_RG`: The Azure resource group where the Quantum Workspace is located. 
   3. `AZURE_QUANTUM_WORKSPACE_NAME`: The name of the Quantum Workspace.
   4. `AZURE_QUANTUM_WORKSPACE_LOCATION`: The region where the Quantum Workspace is located.
5. You **may** [set environment variables][azure auth] to authenticate with Azure. If you do not, a browser will open to the Azure portal to authenticate.
6. Whenever possible, you should prefer using `AzureQuantumComputer.run_batch()` over `Program.write_memory(); AzureQuantumComputer.run()` to run programs which have multiple parameters. Calling `write_memory()` followed by `run()` will still work but will be much slower than running a batch of parameters all at once.


## Examples

### 1. Leveraging Hosted QVM and quilc

With this program, you do not need to run `qvm` nor `quilc` locally in order to leverage them, as they can run through Azure Quantum.

```python
from pyquil_for_azure_quantum import get_qpu, get_qvm
from pyquil.gates import CNOT, MEASURE, H
from pyquil.quil import Program
from pyquil.quilbase import Declare

program = Program(
    Declare("ro", "BIT", 2),
    H(0),
    CNOT(0, 1),
    MEASURE(0, ("ro", 0)),
    MEASURE(1, ("ro", 1)),
).wrap_in_numshots_loop(1000)

qpu = get_qpu("Aspen-11")
qvm = get_qvm()

exe = qpu.compile(program)  # This does not run quilc yet.
results = qpu.run(exe)  # Quilc will run in the cloud before executing the program.
qvm_results = qvm.run(exe)  # This runs the program on QVM in the cloud, not locally.
```

### 2. Running quilc Locally

You can optionally run quilc yourself and disable the use of quilc in the cloud.

```python
from pyquil_for_azure_quantum import get_qpu
from pyquil.gates import CNOT, MEASURE, H
from pyquil.quil import Program
from pyquil.quilbase import Declare


program = Program(
    Declare("ro", "BIT", 2),
    H(0),
    CNOT(0, 1),
    MEASURE(0, ("ro", 0)),
    MEASURE(1, ("ro", 1)),
).wrap_in_numshots_loop(1000)
qpu = get_qpu("Aspen-11")
native_quil = qpu.compiler.quil_to_native_quil(program)  # quilc must be running locally to compile
exe = qpu.compile(native_quil, to_native_gates=False)  # Skip quilc in the cloud
results = qpu.run(exe)
```

### 3. Running Parametrized Circuits in a Batch

When you have a program which should be run across multiple parameters, you can submit all the parameters at once to significantly improve performance.

```python
import numpy as np
from pyquil_for_azure_quantum import get_qpu
from pyquil.gates import MEASURE, RX
from pyquil.quil import Program
from pyquil.quilbase import Declare
from pyquil.quilatom import MemoryReference


program = Program(
    Declare("ro", "BIT", 1),
    Declare("theta", "REAL", 1),
    RX(MemoryReference("theta"), 0),
    MEASURE(0, ("ro", 0)),
).wrap_in_numshots_loop(1000)

qpu = get_qpu("Aspen-11")
compiled = qpu.compile(program)

memory_map = {"theta": [[0.0], [np.pi], [2 * np.pi]]}
results = qpu.run_batch(compiled, memory_map)  # This is a list of results, one for each parameter set.

results_0 = results[0].readout_data["ro"]
results_pi = results[1].readout_data["ro"]
results_2pi = results[2].readout_data["ro"]
```

> Microsoft, Microsoft Azure, and Azure Quantum are trademarks of the Microsoft group of companies. 

[azure-quantum]: https://github.com/microsoft/qdk-python
[pyQuil]: https://pyquil-docs.rigetti.com/en/stable/
[azure auth]: https://docs.microsoft.com/en-us/azure/quantum/optimization-authenticate-service-principal#authenticate-as-the-service-principal
