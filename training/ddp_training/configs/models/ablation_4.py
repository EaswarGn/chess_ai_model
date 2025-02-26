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
        ########### Auxiliary Outputs ##########
        ###############################
        self.move_time_head = nn.Sequential(nn.Linear(self.D_MODEL, 1))
        self.game_length_head = None#nn.Sequential(nn.Linear(self.D_MODEL, 1))
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
            "time_loss_weight": 0.5,
            "result_loss_weight": 0.0,
            "moves_until_end_loss_weight": 1.0,
            "categorical_game_result_loss_weight": 1.0,
        }

        self.move_loss = self.CRITERION
        self.move_time_loss = nn.L1Loss()
        self.moves_until_end_loss = None#nn.L1Loss()
        self.categorical_game_result_loss = nn.CrossEntropyLoss()

