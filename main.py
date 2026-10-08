from config import path
from data import load_and_preprocess_data
from classical import train_classical_svm
from quantum import train_quantum_svm


def main():

    # Load and preprocess data
    X_train, X_test, y_train, y_test = (
        load_and_preprocess_data(path)
    )

    if X_train is None:
        return

    # Classical baseline
    classical_precision, classical_recall = (
        train_classical_svm(
            X_train,
            y_train,
            X_test,
            y_test
        )
    )

    # Quantum model
    quantum_precision, quantum_recall = (
        train_quantum_svm(
            X_train,
            y_train,
            X_test,
            y_test
        )
    )

    # Final comparison
    print("\n" + "=" * 60)
    print("QUANTUM FRAUD DETECTOR")
    print("Classical RBF-SVM vs Quantum Kernel SVM")
    print("=" * 60)

    print(
        f"{'Metric':<20} | "
        f"{'Classical SVM':<18} | "
        f"{'Quantum SVM'}"
    )

    print("-" * 60)

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

    print("=" * 60)

    print("\n[BUSINESS BRIEF]")
    print(
        "• Fraud and legitimate transactions were balanced "
        "before training."
    )
    print(
        "• PCA reduced 30 features to 3 quantum features."
    )
    print(
        "• The quantum model uses a 3-qubit ZZ feature map."
    )
    print(
        "• Recall is important because missed fraud "
        "can represent financial loss."
    )


if __name__ == "__main__":
    main()