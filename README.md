# pyquil-azure-quantum

This library allows you to use [`pyquil`] to run programs on [Azure Quantum](https://azure.microsoft.com/en-us/services/quantum/) against Rigetti targets. Internally, it leverages the [`azure-quantum`] package.

## Usage

Generally, you use [`pyquil`] normally, with a few differences:

1. Instead of `pyquil.get_qc()`, you will use either `pyquil_azure.get_qvm()` or `pyquil_azure.get_qpu()`.
2. You do not need to have `qvm` or `quilc` running in order to run programs through `pyquil_azure`. You may still run them if you wish to run QVM locally instead of passing through Azure or if you wish to precompile your programs (e.g., to inspect the exact Quil that will run).
3. You do not need a QCS account or credentials unless you wish to manually inspect the details of the QPU (e.g., list all qubits).
4. You **must** have these environment variables set:
   1. AZURE_QUANTUM_SUBSCRIPTION_ID: The Azure subscription ID where the Quantum Workspace is located.
   2. AZURE_QUANTUM_WORKSPACE_RG: The Azure resource group where the Quantum Workspace is located. 
   3. AZURE_QUANTUM_WORKSPACE_NAME: The name of the Quantum Workspace.
   4. AZURE_QUANTUM_WORKSPACE_LOCATION: The region where the Quantum Workspace is located.
5. You **may** have these environment variables set—if you do not, a browser will open to the Azure portal to authenticate:
   1. AZURE_CLIENT_ID 
   2. AZURE_CLIENT_SECRET 
   3. AZURE_TENANT_ID
6. Whenever possible, you should prefer using `AzureQuantumComputer.run_batch()` over `Program.write_memory(); AzureQuantumComputer.run()` to run programs which have multiple parameters. Calling `write_memory()` followed by `run()` will still work but will be much slower than running a batch of parameters all at once.

[`azure-quantum`]: https://github.com/microsoft/qdk-python
[`pyquil`]: https://pyquil-docs.rigetti.com/en/stable/
