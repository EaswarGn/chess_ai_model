import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
import time
import argparse
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import math
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import shutil

import sys
from utils import *
from configs import import_config
from criteria_ddp import MultiTaskChessLoss, LabelSmoothedCE
from datasets_ddp import ChunkLoader
from model_ddp import ChessTemporalTransformerEncoder
import numpy as np
import subprocess
import random
import datetime


cudnn.benchmark = False

class HarmoniaTrainer:
    def __init__(self, CONFIG) -> None:
        
        backend='nccl'
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            dist.init_process_group(backend=backend)
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
            print("yes")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )  # CPU isn't really practical here
            self.master_process = True