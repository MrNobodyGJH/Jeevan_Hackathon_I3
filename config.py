# Add the dataset's file path here
path = ''

TARGET_COLUMN = "Class"

QUANTUM_FEATURE_DIM = 3
MAX_SAMPLES_PER_CLASS = 200

TEST_SIZE = 0.25
RANDOM_STATE = 42


# "aer" = local simulation
# "ibm" = real IBM quantum hardware

QUANTUM_BACKEND = "ibm"

IBM_TOKEN = None
IBM_INSTANCE = "open-instance"
IBM_BACKEND_NAME = None
IBM_SHOTS = 512
IBM_MAX_TRAIN_SAMPLES = 20
IBM_MAX_TEST_SAMPLES = 8