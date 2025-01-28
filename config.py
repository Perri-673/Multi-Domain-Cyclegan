import torch

# Set up device (CUDA or CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# General settings
IMG_SIZE = 256  # Size of images for training
IMG_CHANNELS = 3  # Color images (RGB)

# Data Settings
# Number of domains for multi-domain CycleGAN (e.g., healthy, early blight, late blight)
NUM_DOMAINS = 3
DOMAINS = ['healthy', 'early_blight', 'late_blight']

# Path to datasets (adjust according to your folder structure)
DATA_PATH = "path_to_your_data"
TRAIN_DIR = f"data/Train"
VAL_DIR = f"data/Val"

# Training hyperparameters
BATCH_SIZE = 16
LR = 2e-4  # Initial learning rate
LR_DISC = 0.0002  # Learning rate for the discriminator
LR_GEN = 0.0002   # Learning rate for the generator

# Optionally, you can also add other hyperparameters like betas:
BETA1 = 0.5       # Beta1 for Adam optimizer
BETA2 = 0.999     # Beta2 for Adam optimizer
BETA1 = 0.5
BETA2 = 0.999
NUM_EPOCHS = 100
LAMBDA_CYCLE = 10.0  # Cycle consistency loss coefficient
LAMBDA_IDENTITY = 0.5  # Identity loss coefficient
LAMBDA_DOMAIN_CLASSIFICATION = 1.0  # Domain classification loss coefficient

# Model paths for saving and loading checkpoints
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILE = "checkpoint.pth.tar"

# Logging and image saving settings
SAVE_IMAGE_INTERVAL = 200  # Save images every 200 steps
SAVE_MODEL_INTERVAL = 10  # Save model checkpoints every 10 epochs

# Learning rate scheduling
LR_SCHEDULER_TYPE = 'StepLR'  # Options: 'StepLR', 'ExponentialLR', 'CosineAnnealingLR'
STEP_SIZE = 10
GAMMA = 0.1


# Set random seed for reproducibility
SEED = 42  # Add this line for the SEED value
