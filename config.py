import torch

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data
DATA_DIR = "data"
BENCHMARK = 1
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Collocation points
N_COLLOCATION = 1200

# Network
INPUT_DIM = 3  # x, t, u
OUTPUT_DIM = 4  # NO3, DOC, Fe2+, N2
HIDDEN_DIM = 128
NUM_LAYERS = 6

# Training
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS_STAGE1 = 20000  # Transport
EPOCHS_STAGE2 = 20000  # Reaction
EPOCHS_STAGE3 = 10000  # Fine-tune
CHECKPOINT_FREQ = 1000
LOSS_WEIGHT_UPDATE_FREQ = 100

# Optimizer
OPTIMIZER = "Adam"  # Stage 1, 2
OPTIMIZER_STAGE3 = "LBFGS"

# Directories
MODEL_DIR = "models"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"