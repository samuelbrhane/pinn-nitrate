import torch

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data Configuration
DATA_DIR = "data"
BENCHMARK = 1
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Domain Bounds
X_MIN = 0.0
X_MAX = 100.0
T_MIN = 0.0
T_MAX = 100.0
U = 0.5

# Collocation Points
N_COLLOCATION = 1200

# Network Architecture
INPUT_DIM = 3
OUTPUT_DIM = 4
HIDDEN_DIM = 128
NUM_LAYERS = 6

# Training Parameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS_STAGE1 = 5000
EPOCHS_STAGE2 = 5000
EPOCHS_STAGE3 = 2000
CHECKPOINT_FREQ = 1000
LOSS_WEIGHT_UPDATE_FREQ = 500

# Physics Parameters
D = 1.0
K = 0.01

# Optimizer Configuration
OPTIMIZER = "Adam"
OPTIMIZER_STAGE3 = "LBFGS"

# Directory Paths
MODEL_DIR = f"models/benchmark{BENCHMARK}"
RESULTS_DIR = f"results/benchmark{BENCHMARK}"
PLOTS_DIR = f"plots/benchmark{BENCHMARK}"

# Inlet and Boundary Conditions
NO3_INLET = 50.0
DOC_INLET = 200.0
FE2_INLET = 1.0
N2_INLET = 0.0

# Stoichiometric Coefficients
STOICH_NO3 = 1.0
STOICH_DOC = 0.5
STOICH_FE2 = 0.5
STOICH_N2 = 0.5