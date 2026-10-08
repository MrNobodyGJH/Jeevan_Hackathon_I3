import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score
)

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from config import RANDOM_STATE

from quantum import (
    connect_to_ibm,
    create_feature_map,
    compute_ibm_train_kernel,
    compute_ibm_test_kernel
)

from print import print_experiment_start


def select_training_data(
    X_train,
    y_train,
    size
):
    X_small, _, y_small, _ = train_test_split(
        X_train,
        y_train,
        train_size=size,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    return X_small, y_small


def evaluate_classical_model(
    X_train,
    y_train,
    X_test,
    y_test
):
    model = SVC(
        kernel="rbf",
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return precision, recall, accuracy


def evaluate_quantum_model(
    backend,
    feature_map,
    X_train,
    y_train,
    X_test,
    y_test
):
    # Quantum training kernel
    train_matrix = compute_ibm_train_kernel(
        backend,
        feature_map,
        X_train
    )

    # Quantum test kernel
    test_matrix = compute_ibm_test_kernel(
        backend,
        feature_map,
        X_test,
        X_train
    )

    # SVM using the quantum kernel
    model = SVC(
        kernel="precomputed"
    )

    model.fit(
        train_matrix,
        y_train
    )

    predictions = model.predict(
        test_matrix
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return precision, recall, accuracy


def run_training_size_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    training_sizes
):
    print_experiment_start(
        training_sizes
    )

    backend = connect_to_ibm()

    results = []

    for size in training_sizes:

        print(
            "\n" + "-" * 60
        )

        print(
            f"Training size: {size} samples"
        )

        X_small, y_small = select_training_data(
            X_train,
            y_train,
            size
        )

        # Classical model
        print(
            "Running classical SVM..."
        )

        (
            classical_precision,
            classical_recall,
            classical_accuracy
        ) = evaluate_classical_model(
            X_small,
            y_small,
            X_test,
            y_test
        )

        print(
            f"Classical SVM -> "
            f"Precision: {classical_precision:.4f}, "
            f"Recall: {classical_recall:.4f}, "
            f"Accuracy: {classical_accuracy:.4f}"
        )

        # -------------------------------
        # Quantum model
        # -------------------------------

        print(
            "Running quantum SVM..."
        )

        # Create a fresh feature map for each experiment
        feature_map = create_feature_map()

        (
            quantum_precision,
            quantum_recall,
            quantum_accuracy
        ) = evaluate_quantum_model(
            backend,
            feature_map,
            X_small,
            y_small,
            X_test,
            y_test
        )

        print(
            f"Quantum SVM -> "
            f"Precision: {quantum_precision:.4f}, "
            f"Recall: {quantum_recall:.4f}, "
            f"Accuracy: {quantum_accuracy:.4f}"
        )

        results.append({
            "Training Size": size,

            "Classical Precision":
                classical_precision,

            "Quantum Precision":
                quantum_precision,

            "Classical Recall":
                classical_recall,

            "Quantum Recall":
                quantum_recall,

            "Classical Accuracy":
                classical_accuracy,

            "Quantum Accuracy":
                quantum_accuracy
        })

    return pd.DataFrame(results)