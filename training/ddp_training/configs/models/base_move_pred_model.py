import torch
import pathlib
from torch import nn
import multiprocessing as mp
from .utils.levels import TURN, PIECES, UCI_MOVES, BOOL
from .utils.utils import get_lr
from .utils.criteria import LabelSmoothedCE
from .utils.time_controls import time_controls_encoded


#picklable config class for multiprocessing and distributed training on multiple gpus
class CONFIG:
    def __init__(self):
        ###############################
        ############ Name #############
        ###############################
        self.NAME = "base_move_pred_model"
        self.NUM_GPUS = torch.cuda.device_count() #uses all available GPU's on system

        ###############################
        ######### Dataloading #########
        ###############################
        self.BATCH_SIZE = 512
        if self.NUM_GPUS == 0:
            self.NUM_WORKERS = mp.cpu_count()
        else:
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
            "time_controls": len(time_controls_encoded)
        }  # vocabulary sizes
        self.D_MODEL = 512  # size of vectors throughout the transformer model
        self.N_HEADS = 8  # number of heads in the multi-head attention
        self.D_QUERIES = 64  # size of query vectors (and also the size of the key vectors) in the multi-head attention
        self.D_VALUES = 64  # size of value vectors in the multi-head attention
        self.D_INNER = 2048  # an intermediate size in the position-wise FC
        self.N_LAYERS = 6  # number of layers in the Encoder and Decoder
        self.DROPOUT = 0.1  # dropout probability
        self.N_MOVES = 1  # expected maximum length of move sequences in the model, <= MAX_MOVE_SEQUENCE_LENGTH
        self.DISABLE_COMPILATION = False  # disable model compilation?
        self.COMPILATION_MODE = "default"  # mode of model compilation (see torch.compile())
        self.DYNAMIC_COMPILATION = True  # expect tensors with dynamic shapes?
        self.SAMPLING_K = 1  # k in top-k sampling model predictions during play

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
        self.USE_STRICT = False #use strict loading when loading a checkpoint?
        self.CHECKPOINT_PATH = '../../../base_move_pred_model.pt'#../../../averaged_CT-EFT-20.pt'#'../../../1900_step_10000.pt'
        self.VALIDATION_STEPS = 100 #number of validation steps (each step has BATCH_SIZE samples)

        ###############################
        ########### Auxiliary Outputs ##########
        ###############################
        self.move_time_head = None
        self.game_length_head = None
        self.categorical_game_result_head = None
        self.game_result_head = None

        ###############################
        ########### LOSS ##########
        ###############################
        self.CRITERION = LabelSmoothedCE
        self.LOSS_WEIGHTS = {
            "move_loss_weight": 1.0,
            "time_loss_weight": 0.0,
            "result_loss_weight": 0.0,
            "moves_until_end_loss_weight": 0.0,
            "categorical_game_result_loss_weight": 0.0,
        }

        self.move_loss = self.CRITERION
        self.move_time_loss = None
        self.moves_until_end_loss = None
        self.categorical_game_result_loss = None

