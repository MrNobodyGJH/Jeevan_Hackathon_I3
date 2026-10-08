import numpy as np
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler

from config import QUANTUM_FEATURE_DIM


def create_quantum_kernel():
    """Create the Qiskit fidelity quantum kernel."""

    # ZZFeatureMap: encode classical features into qubits
    feature_map = ZZFeatureMap(
        feature_dimension=QUANTUM_FEATURE_DIM,
        reps=2,
        entanglement="linear"
    )

    # AerSimulator: local quantum simulation
    backend = AerSimulator()

    # ISA circuit: transpile for simulator backend
    feature_map_isa = transpile(
        feature_map,
        backend=backend
    )

    sampler = Sampler()

    # Fidelity estimation using Compute-Uncompute
    fidelity = ComputeUncompute(
        sampler=sampler
    )

    quantum_kernel = FidelityQuantumKernel(
        fidelity=fidelity,
        feature_map=feature_map_isa
    )

    return quantum_kernel


def compute_train_kernel(quantum_kernel, X_train):
    """Compute the symmetric training quantum kernel matrix."""

    n_train = len(X_train)

    kernel_matrix = np.zeros(
        (n_train, n_train)
    )

    print("\n[INFO] Computing training quantum kernel...")

    for i in tqdm(
        range(n_train),
        desc="Train Matrix",
        unit="row"
    ):
        # Kernel symmetry: K(x,y) = K(y,x)
        row_values = quantum_kernel.evaluate(
            x_vec=X_train[i:i + 1],
            y_vec=X_train[i:]
        )[0]

        kernel_matrix[i, i:] = row_values
        kernel_matrix[i:, i] = row_values

    return kernel_matrix


def compute_test_kernel(quantum_kernel, X_test, X_train):
    """Compute the test-to-training quantum kernel matrix."""

    n_test = len(X_test)
    n_train = len(X_train)

    kernel_matrix = np.zeros(
        (n_test, n_train)
    )

    print("\n[INFO] Computing test quantum kernel...")

    for i in tqdm(
        range(n_test),
        desc="Test Matrix",
        unit="row"
    ):
        # Test samples are compared against every training sample
        kernel_matrix[i, :] = quantum_kernel.evaluate(
            x_vec=X_test[i:i + 1],
            y_vec=X_train
        )[0]

    return kernel_matrix


def train_quantum_svm(X_train, y_train, X_test, y_test):
    """Train and evaluate an SVM using a quantum kernel."""

    print("\n[INFO] Setting up Quantum Kernel...")

    quantum_kernel = create_quantum_kernel()

    train_matrix = compute_train_kernel(
        quantum_kernel,
        X_train
    )

    test_matrix = compute_test_kernel(
        quantum_kernel,
        X_test,
        X_train
    )

    # Precomputed kernel SVM
    print(
        "\n[INFO] Training SVM "
        "with quantum kernel..."
    )

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