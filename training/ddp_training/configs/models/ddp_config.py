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

NAME = "ddp_config"  # name and identifier for this configuration
NUM_GPUS = 2#, 2, 3, 4, 5, 6, 7]

###############################
######### Dataloading #########
###############################

#DATASET = ChessDatasetFT  # custom PyTorch dataset
BATCH_SIZE = 512  # batch size
NUM_WORKERS = mp.cpu_count()//NUM_GPUS  # number of workers to use for dataloading
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
D_MODEL = 384  # size of vectors throughout the transformer model
N_HEADS = 8 #12  # number of heads in the multi-head attention
D_QUERIES = 64  # size of query vectors (and also the size of the key vectors) in the multi-head attention
D_VALUES = 64  # size of value vectors in the multi-head attention
D_INNER = 4 * D_MODEL  # an intermediate size in the position-wise FC
N_LAYERS = 12  # number of layers in the Encoder and Decoder
DROPOUT = 0.2  # dropout probability
N_MOVES = 1  # expected maximum length of move sequences in the model, <= MAX_MOVE_SEQUENCE_LENGTH
DISABLE_COMPILATION = False  # disable model compilation?
COMPILATION_MODE = "default"  # mode of model compilation (see torch.compile())
DYNAMIC_COMPILATION = True  # expect tensors with dynamic shapes?
SAMPLING_K = 1  # k in top-k sampling model predictions during play
#MODEL = ChessTransformerEncoderFT  # custom PyTorch model to train

###############################
########### Training ##########
###############################

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
OPTIMIZER = torch.optim.Adam  # optimizer
CHECKPOINT_PATH = None


###############################
########### Outputs ##########
###############################
move_time_head = nn.Sequential(
            nn.Linear(D_MODEL, 1),
        )
game_length_head = nn.Sequential(
            nn.Linear(D_MODEL, 1),
        )
categorical_game_result_head = nn.Sequential(
            nn.Linear(D_MODEL, 3),
            nn.Softmax(dim=-1)  # Changed to Softmax to output probabilities
        )
game_result_head = None


###############################
########### LOSS ##########
###############################
CRITERION = LabelSmoothedCE  # training criterion (loss)
LOSS_WEIGHTS = {
    'move_loss_weight': 1.0,
    'move_time_loss_weight': 0.5,
    'game_result_loss_weight': 0.0,
    'moves_until_end_loss_weight': 1.0,
    'categorical_game_result_loss_weight': 1.0
}

move_loss =  CRITERION
move_time_loss= nn.L1Loss(),
#'game_result_loss': nn.L1Loss(),
moves_until_end_loss= nn.L1Loss(),
categorical_game_result_loss= nn.CrossEntropyLoss()



class MODEL_CONFIG:
    def __init__(self):
        ###############################
        ############ Name #############
        ###############################
        self.NAME = "ddp_config"
        self.NUM_GPUS = 2

        ###############################
        ######### Dataloading #########
        ###############################
        self.BATCH_SIZE = 512
        self.NUM_WORKERS = mp.cpu_count() // self.NUM_GPUS
        self.PREFETCH_FACTOR = 2
        self.PIN_MEMORY = False

        ###############################
        ############ Model ############
        ###############################
        self.VOCAB_SIZES = {
            "moves": len(UCI_MOVES),
            "turn": len(TURN),
            "white_kingside_castling_rights": len(BOOL),
            "white_queenside_castling_rights": len(BOOL),
            "black_kingside_castling_rights": len(BOOL),
            "black_queenside_castling_rights": len(BOOL),
            "board_position": len(PIECES),
            "time_controls": len(time_controls_encoded),
        }
        self.D_MODEL = 384
        self.N_HEADS = 8
        self.D_QUERIES = 64
        self.D_VALUES = 64
        self.D_INNER = 4 * self.D_MODEL
        self.N_LAYERS = 12
        self.DROPOUT = 0.2
        self.N_MOVES = 1
        self.DISABLE_COMPILATION = False
        self.COMPILATION_MODE = "default"
        self.DYNAMIC_COMPILATION = True
        self.SAMPLING_K = 1

        ###############################
        ########### Training ##########
        ###############################
        self.BATCHES_PER_STEP = 4
        self.PRINT_FREQUENCY = 1
        self.N_STEPS = 10000
        self.STEPS_PER_EPOCH = 2000
        self.WARMUP_STEPS = 3000
        self.STEP = 1
        self.LR_SCHEDULE = "exp_decay"
        self.LR_DECAY = 0.06
        self.LR = get_lr(
            step=self.STEP,
            d_model=self.D_MODEL,
            warmup_steps=self.WARMUP_STEPS,
            schedule=self.LR_SCHEDULE,
            decay=self.LR_DECAY,
        )
        self.START_EPOCH = 0
        self.BETAS = (0.9, 0.98)
        self.EPSILON = 1e-9
        self.LABEL_SMOOTHING = 0.1
        self.BOARD_STATUS_LENGTH = 70
        self.USE_AMP = True
        self.OPTIMIZER = torch.optim.Adam
        self.CHECKPOINT_PATH = None

        ###############################
        ########### Outputs ##########
        ###############################
        self.move_time_head = nn.Sequential(nn.Linear(self.D_MODEL, 1))
        self.game_length_head = nn.Sequential(nn.Linear(self.D_MODEL, 1))
        self.categorical_game_result_head = nn.Sequential(
            nn.Linear(self.D_MODEL, 3),
            nn.Softmax(dim=-1),
        )
        self.game_result_head = None

        ###############################
        ########### LOSS ##########
        ###############################
        self.CRITERION = LabelSmoothedCE
        self.LOSS_WEIGHTS = {
            "move_loss_weight": 1.0,
            "move_time_loss_weight": 0.5,
            "game_result_loss_weight": 0.0,
            "moves_until_end_loss_weight": 1.0,
            "categorical_game_result_loss_weight": 1.0,
        }

        self.move_loss = self.CRITERION
        self.move_time_loss = nn.L1Loss()
        self.moves_until_end_loss = nn.L1Loss()
        self.categorical_game_result_loss = nn.CrossEntropyLoss()

