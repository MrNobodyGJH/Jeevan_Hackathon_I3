import matplotlib.pyplot as plt


def plot_precision(results):
    """Plot precision against training size."""

    plt.figure(figsize=(8, 5))

    plt.plot(
        results["Training Size"],
        results["Classical Precision"],
        marker="o",
        label="Classical SVM"
    )

    plt.plot(
        results["Training Size"],
        results["Quantum Precision"],
        marker="o",
        label="Quantum SVM"
    )

    plt.xlabel("Training Size")
    plt.ylabel("Precision")
    plt.title("Precision vs Training Size")

    plt.xticks(
        results["Training Size"]
    )

    plt.ylim(0, 1.05)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "precision_comparison.png",
        dpi=300
    )

    plt.show()


def plot_recall(results):
    """Plot recall against training size."""

    plt.figure(figsize=(8, 5))

    plt.plot(
        results["Training Size"],
        results["Classical Recall"],
        marker="o",
        label="Classical SVM"
    )

    plt.plot(
        results["Training Size"],
        results["Quantum Recall"],
        marker="o",
        label="Quantum SVM"
    )

    plt.xlabel("Training Size")
    plt.ylabel("Recall")
    plt.title("Recall vs Training Size")

    plt.xticks(
        results["Training Size"]
    )

    plt.ylim(0, 1.05)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "recall_comparison.png",
        dpi=300
    )

    plt.show()


def plot_accuracy(results):
    """Plot accuracy against training size."""

    plt.figure(figsize=(8, 5))

    plt.plot(
        results["Training Size"],
        results["Classical Accuracy"],
        marker="o",
        label="Classical SVM"
    )

    plt.plot(
        results["Training Size"],
        results["Quantum Accuracy"],
        marker="o",
        label="Quantum SVM"
    )

    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Training Size")

    plt.xticks(
        results["Training Size"]
    )

    plt.ylim(0, 1.05)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "accuracy_comparison.png",
        dpi=300
    )

    plt.show()