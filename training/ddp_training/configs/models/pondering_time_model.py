import torch
import pathlib
from torch import nn
import multiprocessing as mp
from .utils.levels import TURN, PIECES, UCI_MOVES, BOOL
from .utils.utils import get_lr
from .utils.criteria import LabelSmoothedCE, FocalLoss
from .utils.time_controls import time_controls_encoded


#picklable config class for multiprocessing and distributed training on multiple gpus
class CONFIG:
    def __init__(self):
        ###############################
        ############ Name #############
        ###############################
        self.NAME = "pondering_time_model"
        self.NUM_GPUS = torch.cuda.device_count()

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
        self.USE_UPLOAD = False #upload checkpoints to huggingface?
        self.BATCHES_PER_STEP = 4
        self.PRINT_FREQUENCY = 10
        self.N_STEPS = None
        self.STEPS_PER_EPOCH = 1000
        self.WARMUP_STEPS = 3000
        self.STEP = None #the step to start training at, if None then step will start at 1 even after loading from checkpoint
        self.LR_SCHEDULE = "exp_decay"
        self.LR_DECAY = 0.06
        self.LR = get_lr(
            step=1,
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
        self.USE_STRICT = True #use strict loading when loading a checkpoint?
        self.CHECKPOINT_PATH = '../../../full_trained_model.pt'#'../../../pondering_time_step_15000.pt'#
        self.VALIDATION_STEPS = 100 #number of validation steps (each step has BATCH_SIZE samples)

        ###############################
        ########### Auxiliary Outputs ##########
        ###############################
        self.move_time_head = nn.Sequential(nn.Linear(self.D_MODEL, 1))
        self.game_length_head = None#nn.Sequential(nn.Linear(self.D_MODEL, 1))
        self.categorical_game_result_head = None 
        """nn.Sequential(
            nn.Linear(self.D_MODEL, 3)
        )"""
        self.game_result_head = None

        ###############################
        ########### LOSS ##########
        ###############################
        self.CRITERION = LabelSmoothedCE
        self.LOSS_WEIGHTS = {
            "move_loss_weight": 1.0,
            "time_loss_weight": 1.0,
            "result_loss_weight": 0.0,
            "moves_until_end_loss_weight": 1.0,
            "categorical_game_result_loss_weight": 1.0,
        }

        
        
        self.move_loss = None #self.CRITERION
        self.move_time_loss = nn.L1Loss()
        self.moves_until_end_loss = None #nn.L1Loss()
        self.categorical_game_result_loss = None #nn.CrossEntropyLoss

