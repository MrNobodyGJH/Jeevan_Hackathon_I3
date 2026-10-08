import numpy as np

from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator

from qiskit_machine_learning.kernels import (
    FidelityQuantumKernel
)

from qiskit_machine_learning.state_fidelities import (
    ComputeUncompute
)

from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    SamplerV2
)

from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager
)

from config import (
    QUANTUM_FEATURE_DIM,
    QUANTUM_BACKEND,
    IBM_TOKEN,
    IBM_INSTANCE,
    IBM_BACKEND_NAME,
    IBM_SHOTS,
    IBM_MAX_TRAIN_SAMPLES,
    IBM_MAX_TEST_SAMPLES,
    RANDOM_STATE
)

from print import (
    print_quantum_start,
    print_hardware_dataset,
    print_backend,
    print_training_kernel,
    print_testing_kernel,
    print_quantum_svm
)

# AER Quantum kernel
def create_aer_kernel():
    print_quantum_start("Aer simulator")

    feature_map = ZZFeatureMap(
        feature_dimension=QUANTUM_FEATURE_DIM,
        reps=2,
        entanglement="linear",
        parameter_prefix="feature"
    )

    backend = AerSimulator()

    feature_map = transpile(
        feature_map,
        backend=backend
    )

    sampler = SamplerV2(
        mode=backend
    )

    fidelity = ComputeUncompute(
        sampler=sampler
    )

    kernel = FidelityQuantumKernel(
        fidelity=fidelity,
        feature_map=feature_map
    )

    return kernel

# IBM CONNECTION
def connect_to_ibm():
    if IBM_TOKEN:
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=IBM_TOKEN,
            instance=IBM_INSTANCE
        )
    else:
        service = QiskitRuntimeService(
            instance=IBM_INSTANCE
        )

    if IBM_BACKEND_NAME:
        backend = service.backend(
            IBM_BACKEND_NAME
        )
    else:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=QUANTUM_FEATURE_DIM
        )

    print_backend(
        backend.name,
        backend.status().pending_jobs
    )

    return backend

# FEATURE MAP
def create_feature_map():
    return ZZFeatureMap(
        feature_dimension=QUANTUM_FEATURE_DIM,
        reps=2,
        entanglement="linear",
        parameter_prefix="feature"
    )

# IBM FIDELITY CIRCUIT
def create_fidelity_circuit(
    feature_map,
    x,
    y
):
    # Binding first data point
    state_x = feature_map.assign_parameters(
        x,
        inplace=False
    )

    # Binding second data point
    state_y = feature_map.assign_parameters(
        y,
        inplace=False
    )

    fidelity_circuit = QuantumCircuit(
        QUANTUM_FEATURE_DIM
    )

    fidelity_circuit.compose(
        state_x,
        inplace=True
    )

    fidelity_circuit.compose(
        state_y.inverse(),
        inplace=True
    )

    fidelity_circuit.measure_all()

    return fidelity_circuit

# READ IBM FIDELITY
def get_zero_probability(result):
    counts = result.data.meas.get_counts()

    zero_state = "0" * QUANTUM_FEATURE_DIM

    return counts.get(
        zero_state,
        0
    ) / sum(counts.values())

# RUN IBM CIRCUITS
def run_ibm_fidelity_circuits(
    backend,
    circuits
):
    # Hardware-aware transpilation
    # Every circuit must match the QPU's ISA
    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=1
    )

    isa_circuits = pass_manager.run(
        circuits
    )

    # SamplerV2: submit ISA circuits to IBM hardware
    sampler = SamplerV2(
        mode=backend,
        options={
            "default_shots": IBM_SHOTS
        }
    )

    job = sampler.run(
        isa_circuits
    )

    print(
        f"IBM job submitted: {job.job_id()}"
    )

    result = job.result()

    fidelities = []

    for circuit_result in result:
        fidelities.append(
            get_zero_probability(
                circuit_result
            )
        )

    return fidelities

# IBM TRAINING KERNEL
def compute_ibm_train_kernel(
    backend,
    feature_map,
    X_train
):
    n_train = len(X_train)

    kernel_matrix = np.zeros(
        (n_train, n_train)
    )

    print_training_kernel()

    circuits = []
    positions = []
    for i in range(n_train):

        kernel_matrix[i, i] = 1.0

        # Only calculate one half because: K(x, y) = K(y, x)
        for j in range(i + 1, n_train):

            circuit = create_fidelity_circuit(
                feature_map,
                X_train[i],
                X_train[j]
            )

            circuits.append(circuit)
            positions.append((i, j))

    print(
        f"Preparing {len(circuits)} quantum circuits..."
    )

    fidelities = run_ibm_fidelity_circuits(
        backend,
        circuits
    )

    for (i, j), fidelity in zip(
        positions,
        fidelities
    ):
        kernel_matrix[i, j] = fidelity
        kernel_matrix[j, i] = fidelity

    return kernel_matrix

# IBM TEST KERNEL
def compute_ibm_test_kernel(
    backend,
    feature_map,
    X_test,
    X_train
):
    n_test = len(X_test)
    n_train = len(X_train)

    kernel_matrix = np.zeros(
        (n_test, n_train)
    )

    print_testing_kernel()

    circuits = []
    positions = []

    for i in range(n_test):

        for j in range(n_train):

            circuit = create_fidelity_circuit(
                feature_map,
                X_test[i],
                X_train[j]
            )

            circuits.append(circuit)
            positions.append((i, j))

    print(
        f"Preparing {len(circuits)} quantum circuits..."
    )

    fidelities = run_ibm_fidelity_circuits(
        backend,
        circuits
    )

    for (i, j), fidelity in zip(
        positions,
        fidelities
    ):
        kernel_matrix[i, j] = fidelity

    return kernel_matrix

# HARDWARE DATASET SIZE
def reduce_hardware_dataset(
    X_train,
    y_train,
    X_test,
    y_test
):
    if QUANTUM_BACKEND.lower() != "ibm":
        return (
            X_train,
            y_train,
            X_test,
            y_test
        )

    train_size = min(
        IBM_MAX_TRAIN_SAMPLES,
        len(X_train)
    )

    test_size = min(
        IBM_MAX_TEST_SAMPLES,
        len(X_test)
    )

    X_train_small, _, y_train_small, _ = train_test_split(
        X_train,
        y_train,
        train_size=train_size,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    X_test_small, _, y_test_small, _ = train_test_split(
        X_test,
        y_test,
        train_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y_test
    )

    print_hardware_dataset(
        len(X_train_small),
        len(X_test_small)
    )

    return (
        X_train_small,
        y_train_small,
        X_test_small,
        y_test_small
    )

# QUANTUM MODEL
def train_quantum_svm(
    X_train,
    y_train,
    X_test,
    y_test
):
    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = reduce_hardware_dataset(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # IBM HARDWARE
    if QUANTUM_BACKEND.lower() == "ibm":

        backend = connect_to_ibm()

        feature_map = create_feature_map()

        # Training kernel
        train_matrix = compute_ibm_train_kernel(
            backend,
            feature_map,
            X_train
        )

        # Testing kernel
        test_matrix = compute_ibm_test_kernel(
            backend,
            feature_map,
            X_test,
            X_train
        )

    # AER SIMULATION
    else:
        quantum_kernel = create_aer_kernel()

        train_matrix = compute_aer_train_kernel(
            quantum_kernel,
            X_train
        )

        test_matrix = compute_aer_test_kernel(
            quantum_kernel,
            X_test,
            X_train
        )

    # CLASSICAL SVM USING QUANTUM KERNEL
    print_quantum_svm()

    # Precomputed kernel SVM
    model = SVC(
        kernel="precomputed"
    )

    model.fit(
        train_matrix,
        y_train
    )

    y_pred = model.predict(
        test_matrix
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    return precision, recall

# AER TRAINING KERNEL
def compute_aer_train_kernel(
    quantum_kernel,
    X_train
):
    n_train = len(X_train)

    kernel_matrix = np.zeros(
        (n_train, n_train)
    )

    print_training_kernel()

    for i in tqdm(
        range(n_train),
        desc="Training kernel",
        unit="row"
    ):
        row_values = quantum_kernel.evaluate(
            x_vec=X_train[i:i + 1],
            y_vec=X_train[i:]
        )[0]

        kernel_matrix[i, i:] = row_values
        kernel_matrix[i:, i] = row_values

    return kernel_matrix

# AER TESTING KERNEL
def compute_aer_test_kernel(
    quantum_kernel,
    X_test,
    X_train
):
    n_test = len(X_test)
    n_train = len(X_train)

    kernel_matrix = np.zeros(
        (n_test, n_train)
    )

    print_testing_kernel()

    for i in tqdm(
        range(n_test),
        desc="Testing kernel",
        unit="row"
    ):
        kernel_matrix[i, :] = quantum_kernel.evaluate(
            x_vec=X_test[i:i + 1],
            y_vec=X_train
        )[0]

    return kernel_matrix