import torch

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data
DATA_DIR = "data"
BENCHMARK = 1
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Spatial and Temporal Bounds
X_MIN = 0.0
X_MAX = 100.0
T_MIN = 0.0
T_MAX = 100.0
U = 0.5

# Collocation points
N_COLLOCATION = 1200

# Network
INPUT_DIM = 3
OUTPUT_DIM = 4
HIDDEN_DIM = 128
NUM_LAYERS = 6

# Training
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS_STAGE1 = 5000
EPOCHS_STAGE2 = 5000
EPOCHS_STAGE3 = 2000
CHECKPOINT_FREQ = 1000
LOSS_WEIGHT_UPDATE_FREQ = 500

# Physics parameters (from PHREEQC)
D = 1.0
K = 0.01

# Optimizer
OPTIMIZER = "Adam"
OPTIMIZER_STAGE3 = "LBFGS"

# Directories
MODEL_DIR = f"models/benchmark{BENCHMARK}"
RESULTS_DIR = f"results/benchmark{BENCHMARK}"
PLOTS_DIR = f"plots/benchmark{BENCHMARK}"

# Initial and Boundary Conditions (from SOLUTION 0)
NO3_INLET = 50.0
DOC_INLET = 200.0
FE2_INLET = 1.0
N2_INLET = 0.0

# Stoichiometric coefficients
STOICH_NO3 = 1.0
STOICH_DOC = 0.5
STOICH_FE2 = 0.5
STOICH_N2 = 0.5


# Benchmark 1: Fe2+ is constant → weight = 0 (ignore)
# Other benchmarks: Fe2+ varies → weight = 1.0 (learn)
if BENCHMARK == 1:
    SPECIES_WEIGHTS = {
        'NO3': 1.0,
        'DOC': 1.0,
        'Fe2': 0.0,   # Constant for Benchmark 1
        'N2': 1.0
    }
else:
    SPECIES_WEIGHTS = {
        'NO3': 1.0,
        'DOC': 1.0,
        'Fe2': 1.0,
        'N2': 1.0
    }

# Normalize species by their inlet concentrations
SPECIES_SCALES = {
    'NO3': 1.0 / NO3_INLET,
    'DOC': 1.0 / DOC_INLET,
    'Fe2': 1.0 / FE2_INLET,
    'N2': 1.0 / 50.0  # Max N2 production
}