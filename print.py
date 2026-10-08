def print_dataset_loaded(rows):
    print("Dataset loaded successfully.")
    print(f"Using {rows} balanced transactions for the experiment.")


def print_pca(original_features, quantum_features):
    print(
        f"Reduced the {original_features} features "
        f"to {quantum_features} features using PCA."
    )


def print_classical_start():
    print("\nTraining the classical SVM...")


def print_quantum_start(backend):
    print("\nTraining the quantum SVM...")
    print(f"Quantum backend: {backend}")


def print_hardware_dataset(train_size, test_size):
    print(
        f"Using {train_size} training samples "
        f"and {test_size} test samples on IBM hardware."
    )


def print_backend(backend, pending_jobs):
    print(f"Connected to IBM quantum computer: {backend}")
    print(f"Jobs waiting on the computer: {pending_jobs}")


def print_training_kernel():
    print("\nCalculating the training quantum kernel...")


def print_testing_kernel():
    print("\nCalculating the testing quantum kernel...")


def print_quantum_svm():
    print("\nTraining the SVM using the quantum kernel...")


def print_results(
    classical_precision,
    classical_recall,
    quantum_precision,
    quantum_recall,
    backend
):
    print("\n" + "=" * 65)
    print("QUANTUM FRAUD DETECTION")
    print("=" * 65)

    if backend.lower() == "ibm":
        backend_name = "IBM Quantum Hardware"
    else:
        backend_name = "Qiskit Aer Simulator"

    print(f"Quantum backend: {backend_name}")

    print("-" * 65)

    print(
        f"{'Metric':<20} | "
        f"{'Classical SVM':<18} | "
        f"{'Quantum SVM'}"
    )

    print("-" * 65)

    print(
        f"{'Precision':<20} | "
        f"{classical_precision:<18.4f} | "
        f"{quantum_precision:.4f}"
    )

    print(
        f"{'Recall':<20} | "
        f"{classical_recall:<18.4f} | "
        f"{quantum_recall:.4f}"
    )

    print("=" * 65)

    print("\nSummary")
    print("The dataset was balanced before training.")
    print("PCA reduced the original features to three quantum features.")
    print("The classical model uses an RBF SVM.")
    print("The quantum model uses a fidelity-based quantum kernel.")

    if backend.lower() == "ibm":
        print("The quantum kernel was evaluated on IBM quantum hardware.")
    else:
        print("The quantum kernel was evaluated using the Aer simulator.")

def print_experiment_start(training_sizes):
    print("\n" + "=" * 65)
    print("TRAINING SIZE EXPERIMENT")
    print("=" * 65)

    print(
        "The models will be tested with "
        f"{len(training_sizes)} different training sizes."
    )

    print(
        f"Training sizes: {training_sizes}"
    )

    print(
        "The same test set will be used for every size."
    )