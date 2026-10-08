from config import path

from data import (
    load_and_preprocess_data
)

from experiment import (
    run_training_size_experiment
)

from plots import (
    plot_precision,
    plot_recall,
    plot_accuracy
)


def main():

    # Load and prepare the dataset
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_and_preprocess_data(path)

    if X_train is None:
        return

    # Four gradually increasing training sizes
    training_sizes = [
        5,
        10,
        15,
        20
    ]

    results = run_training_size_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        training_sizes
    )

    # Print the complete comparison
    print("\n" + "=" * 85)
    print("TRAINING SIZE COMPARISON")
    print("=" * 85)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("=" * 85)

    # Create the three metric plots
    plot_precision(results)
    plot_recall(results)
    plot_accuracy(results)

    # Save the numerical results
    results.to_csv(
        "training_size_results.csv",
        index=False
    )

    print(
        "\nThe plots and results table have been saved."
    )


if __name__ == "__main__":
    main()