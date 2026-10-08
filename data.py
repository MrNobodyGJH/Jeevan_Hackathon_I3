import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample

from config import (
    TARGET_COLUMN,
    QUANTUM_FEATURE_DIM,
    MAX_SAMPLES_PER_CLASS,
    TEST_SIZE,
    RANDOM_STATE
)

from print import (
    print_dataset_loaded,
    print_pca
)

def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("The dataset could not be found.")
        print("Please check the path in config.py.")
        return None, None, None, None

    fraud = df[df[TARGET_COLUMN] == 1]
    legit = df[df[TARGET_COLUMN] == 0]

    # Class balancing
    n_samples = min(
        len(fraud),
        MAX_SAMPLES_PER_CLASS
    )

    fraud_sampled = resample(
        fraud,
        replace=False,
        n_samples=n_samples,
        random_state=RANDOM_STATE
    )

    legit_sampled = resample(
        legit,
        replace=False,
        n_samples=n_samples,
        random_state=RANDOM_STATE
    )

    df_balanced = pd.concat(
        [legit_sampled, fraud_sampled]
    )

    # Shuffling
    df_balanced = df_balanced.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print_dataset_loaded(
        len(df_balanced)
    )

    X = df_balanced.drop(
        columns=[TARGET_COLUMN]
    )

    y = df_balanced[TARGET_COLUMN].values

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Standardization
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # PCA
    pca = PCA(
        n_components=QUANTUM_FEATURE_DIM
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    X_test_pca = pca.transform(
        X_test_scaled
    )

    print_pca(
        X.shape[1],
        QUANTUM_FEATURE_DIM
    )

    return (
        X_train_pca,
        X_test_pca,
        y_train,
        y_test
    )