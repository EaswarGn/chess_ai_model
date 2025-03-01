import torch
import pathlib
from torch import nn
import multiprocessing as mp
from .utils.levels import TURN, PIECES, UCI_MOVES, BOOL
from .utils.utils import get_lr
from .utils.criteria import LabelSmoothedCE
from .utils.time_controls import time_controls_encoded


###############################
############ Name #############
###############################

NAME = "ablation_2"  # name and identifier for this configuration
GPU_ID = 0

###############################
######### Dataloading #########
###############################

#DATASET = ChessDatasetFT  # custom PyTorch dataset
BATCH_SIZE = 512  # batch size
NUM_WORKERS = mp.cpu_count()  # number of workers to use for dataloading
PREFETCH_FACTOR = 2  # number of batches to prefetch per worker
PIN_MEMORY = False  # pin to GPU memory when dataloading?

###############################
############ Model ############
###############################

VOCAB_SIZES = {
    "moves": len(UCI_MOVES),
    "turn": len(TURN),
    "white_kingside_castling_rights": len(BOOL),
    "white_queenside_castling_rights": len(BOOL),
    "black_kingside_castling_rights": len(BOOL),
    "black_queenside_castling_rights": len(BOOL),
    "board_position": len(PIECES),
    "time_controls": len(time_controls_encoded)
}  # vocabulary sizes
D_MODEL = 512  # size of vectors throughout the transformer model
N_HEADS = 8  # number of heads in the multi-head attention
D_QUERIES = 64  # size of query vectors (and also the size of the key vectors) in the multi-head attention
D_VALUES = 64  # size of value vectors in the multi-head attention
D_INNER = 2048  # an intermediate size in the position-wise FC
N_LAYERS = 6  # number of layers in the Encoder and Decoder
DROPOUT = 0.1  # dropout probability
N_MOVES = 1  # expected maximum length of move sequences in the model, <= MAX_MOVE_SEQUENCE_LENGTH
DISABLE_COMPILATION = False  # disable model compilation?
COMPILATION_MODE = "default"  # mode of model compilation (see torch.compile())
DYNAMIC_COMPILATION = True  # expect tensors with dynamic shapes?
SAMPLING_K = 1  # k in top-k sampling model predictions during play
OUTPUTS = {
    'from_squares': nn.Linear(D_MODEL, 1),
    'to_squares': nn.Linear(D_MODEL, 1),
    'game_result': None,
    'move_time': None, 
    'moves_until_end': nn.Sequential(
        nn.Linear(D_MODEL, 1),
    ),
    'categorical_game_result': nn.Sequential(
        nn.Linear(D_MODEL, 3),
        nn.Softmax(dim=-1)  # Changed to Softmax to output probabilities
    )
}
#MODEL = ChessTransformerEncoderFT  # custom PyTorch model to train

###############################
########### Training ##########
###############################

USE_STRICT = False #use strict loading when loading weights into model?
BATCHES_PER_STEP = (
    4  # perform a training step, i.e. update parameters, once every so many batches
)
PRINT_FREQUENCY = 1  # print status once every so many steps
N_STEPS = 10000  # number of training steps
STEPS_PER_EPOCH = 2000
WARMUP_STEPS = 3000  # number of warmup steps where learning rate is increased linearly; twice the value in the paper, as in the official transformer repo.
STEP = 1  # the step number, start from 1 to prevent math error in the 'LR' line
LR_SCHEDULE = "exp_decay"  # the learning rate schedule; see utils.py for learning rate schedule
LR_DECAY = 0.06  # the decay rate for 'exp_decay' schedule
LR = get_lr(
    step=STEP,
    d_model=D_MODEL,
    warmup_steps=WARMUP_STEPS,
    schedule=LR_SCHEDULE,
    decay=LR_DECAY,
)  # see utils.py for learning rate schedule
START_EPOCH = 0  # start at this epoch
BETAS = (0.9, 0.98)  # beta coefficients in the Adam optimizer
EPSILON = 1e-9  # epsilon term in the Adam optimizer
LABEL_SMOOTHING = 0.1  # label smoothing co-efficient in the Cross Entropy loss
BOARD_STATUS_LENGTH = 70  # total length of input sequence
USE_AMP = True  # use automatic mixed precision training?
CRITERION = LabelSmoothedCE  # training criterion (loss)
LOSS_WEIGHTS = {
    'move_loss_weight': 1.0,
    'move_time_loss_weight': 1.0,
    'game_result_loss_weight': 1.0,
    'moves_until_end_loss_weight': 1.0,
    'categorical_game_result_loss_weight': 1.0
}
LOSSES = {
    'move_loss': CRITERION(
        eps=LABEL_SMOOTHING, n_predictions=N_MOVES
    ),
    #'move_time_loss': nn.HuberLoss(),
    #'game_result_loss': nn.HuberLoss(),
    'moves_until_end_loss': nn.HuberLoss(),
    'categorical_game_result_loss': nn.CrossEntropyLoss()
}
OPTIMIZER = torch.optim.Adam  # optimizer
CHECKPOINT_PATH = None