# config_class.py
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from copy import deepcopy

@dataclass
class ChessConfig:
    NAME: str
    GPU_ID: List[int]
    BATCH_SIZE: int
    NUM_WORKERS: int
    PREFETCH_FACTOR: int
    PIN_MEMORY: bool
    D_MODEL: int
    N_HEADS: int
    D_QUERIES: int
    D_VALUES: int
    D_INNER: int
    N_LAYERS: int
    DROPOUT: float
    N_MOVES: int
    DISABLE_COMPILATION: bool
    COMPILATION_MODE: str
    DYNAMIC_COMPILATION: bool
    BATCHES_PER_STEP: int
    N_STEPS: int
    STEPS_PER_EPOCH: int
    WARMUP_STEPS: int
    LR_SCHEDULE: str
    LR_DECAY: float
    LR: float
    BETAS: tuple
    EPSILON: float
    LABEL_SMOOTHING: float
    USE_AMP: bool
    LOSS_WEIGHTS: Dict[str, float]
    CHECKPOINT_PATH: Optional[str]
    PRINT_FREQUENCY: int
    VOCAB_SIZES: Dict[str, int]

def convert_config_to_picklable(config):
    """Convert the existing config object to a ChessConfig instance"""
    config_dict = {
        'NAME': config.NAME,
        'GPU_ID': config.GPU_ID,
        'BATCH_SIZE': config.BATCH_SIZE,
        'NUM_WORKERS': config.NUM_WORKERS,
        'PREFETCH_FACTOR': config.PREFETCH_FACTOR,
        'PIN_MEMORY': config.PIN_MEMORY,
        'D_MODEL': config.D_MODEL,
        'N_HEADS': config.N_HEADS,
        'D_QUERIES': config.D_QUERIES,
        'D_VALUES': config.D_VALUES,
        'D_INNER': config.D_INNER,
        'N_LAYERS': config.N_LAYERS,
        'DROPOUT': config.DROPOUT,
        'N_MOVES': config.N_MOVES,
        'DISABLE_COMPILATION': config.DISABLE_COMPILATION,
        'COMPILATION_MODE': config.COMPILATION_MODE,
        'DYNAMIC_COMPILATION': config.DYNAMIC_COMPILATION,
        'BATCHES_PER_STEP': config.BATCHES_PER_STEP,
        'N_STEPS': config.N_STEPS,
        'STEPS_PER_EPOCH': config.STEPS_PER_EPOCH,
        'WARMUP_STEPS': config.WARMUP_STEPS,
        'LR_SCHEDULE': config.LR_SCHEDULE,
        'LR_DECAY': config.LR_DECAY,
        'LR': config.LR,
        'BETAS': config.BETAS,
        'EPSILON': config.EPSILON,
        'LABEL_SMOOTHING': config.LABEL_SMOOTHING,
        'USE_AMP': config.USE_AMP,
        'LOSS_WEIGHTS': config.LOSS_WEIGHTS,
        'CHECKPOINT_PATH': config.CHECKPOINT_PATH,
        'PRINT_FREQUENCY': config.PRINT_FREQUENCY,
        'VOCAB_SIZES': config.VOCAB_SIZES,
    }
    return ChessConfig(**config_dict)