from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score

from config import RANDOM_STATE


def train_classical_svm(X_train, y_train, X_test, y_test):
    """Train and evaluate the classical RBF SVM."""

    print("\n[INFO] Training Classical SVM...")

    # RBF kernel
    model = SVC(
        kernel="rbf",
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

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