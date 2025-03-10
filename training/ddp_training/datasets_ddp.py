import os
import json
import torch
import argparse
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import time
import zstandard as zstd
import torch.distributed as dist
import struct

from configs import import_config
from time_controls import time_controls_encoded
import numpy as np
import multiprocessing as mp
import pickle

# Define the record dtype
record_dtype = np.dtype([
    ("turn", np.int8),
    ("white_kingside_castling_rights", np.int8),
    ("white_queenside_castling_rights", np.int8),
    ("black_kingside_castling_rights", np.int8),
    ("black_queenside_castling_rights", np.int8),
    ("board_position", np.int8, (64,)),
    ("from_square", np.int8),
    ("to_square", np.int8),
    ("length", np.int8),
    ("phase", np.int8),
    ("result", np.int8),
    ("categorical_result", np.int8),
    ("base_time", np.int16),
    ("increment_time", np.int16),
    ("white_remaining_time", np.float16),
    ("black_remaining_time", np.float16),
    ("white_rating", np.int16),
    ("black_rating", np.int16),
    ("time_spent_on_move", np.float16),
    ("move_number", np.int16),
    ("num_legal_moves", np.int16),
    ("white_material_value", np.int16),
    ("black_material_value", np.int16),
    ("material_difference", np.int16),
    ("moves_until_end", np.float16)
])
    
    
from pathlib import Path
def get_all_record_files(directory: str):
    return [str(file) for file in Path(directory).rglob("*") if file.is_file()]


class ChunkLoader(IterableDataset):
    def __init__(self,
                 file_list,
                 record_dtype,
                 rank,
                 world_size,
                 is_val,
                 use_low_time,
                 min_full_move_number=None,
                 max_full_move_number=None
        ):
        self.file_list = file_list
        self.record_dtype = record_dtype
        self.record_size = record_dtype.itemsize
        self.fmt = "<5b64b6b2h2f2hf5hf"
        self.record_size = 109  #struct.calcsize(self.fmt)
        self.length = len(self.file_list) * self.get_chunk_size()

        # Get rank and world size for distributed training
        self.rank = rank
        self.world_size = world_size
        self.is_val = is_val
        self.use_low_time = use_low_time
        self.min_full_move_number = min_full_move_number
        self.max_full_move_number = max_full_move_number

    def get_chunk_size(self):
        with open(self.file_list[0], "rb") as f:
            dctx = zstd.ZstdDecompressor()
            decompressed = dctx.decompress(f.read())
            num_dicts = len(decompressed) // self.record_size
            return num_dicts

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Shard dataset across GPUs
        files = self.file_list[self.rank::self.world_size]

        # Further shard among DataLoader workers
        files = files[worker_id::num_workers]

        for filename in files:
            with open(filename, "rb") as f:
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.decompress(f.read())
                num_dicts = len(decompressed) // self.record_size
                
                for i in range(num_dicts):
                    offset = i * self.record_size
                    record_bytes = decompressed[offset: offset + self.record_size]
                    unpacked = struct.unpack(self.fmt, record_bytes)
                    record = {}
                    idx = 0
                    record["turn"] = unpacked[idx]; idx += 1
                    record["white_kingside_castling_rights"] = unpacked[idx]; idx += 1
                    record["white_queenside_castling_rights"] = unpacked[idx]; idx += 1
                    record["black_kingside_castling_rights"] = unpacked[idx]; idx += 1
                    record["black_queenside_castling_rights"] = unpacked[idx]; idx += 1
                    record["board_position"] = list(unpacked[idx: idx+64]); idx += 64
                    record["from_square"] = unpacked[idx]; idx += 1
                    record["to_square"] = unpacked[idx]; idx += 1
                    record["length"] = unpacked[idx]; idx += 1
                    record["phase"] = unpacked[idx]; idx += 1
                    record["result"] = unpacked[idx]; idx += 1
                    record["categorical_result"] = unpacked[idx]; idx += 1
                    record["base_time"] = unpacked[idx]; idx += 1
                    record["increment_time"] = unpacked[idx]; idx += 1
                    record["white_remaining_time"] = unpacked[idx]; idx += 1
                    record["black_remaining_time"] = unpacked[idx]; idx += 1
                    record["white_rating"] = unpacked[idx]; idx += 1
                    record["black_rating"] = unpacked[idx]; idx += 1
                    record["time_spent_on_move"] = unpacked[idx]; idx += 1
                    record["move_number"] = unpacked[idx]; idx += 1
                    record["num_legal_moves"] = unpacked[idx]; idx += 1
                    record["white_material_value"] = unpacked[idx]; idx += 1
                    record["black_material_value"] = unpacked[idx]; idx += 1
                    record["material_difference"] = unpacked[idx]; idx += 1
                    record["moves_until_end"] = unpacked[idx]; idx += 1
                    
                    if int(record['move_number']) <= 8:
                        record["moves_until_end"] = 35

    
                    try:
                        base_time = record["base_time"]
                        increment_time = record["increment_time"]
                        
                        if int(increment_time)>0 and int(record["move_number"])!=0:
                            record["time_spent_on_move"] = record["time_spent_on_move"] + int(increment_time)
                        if int(record["time_spent_on_move"])<1:
                            record["time_spent_on_move"] = 0
                        
                        time_control = f'{base_time}+{increment_time}'
                        time_control = torch.LongTensor([5])
                    except KeyError:
                        pass

                    if int(record["turn"]) == 0:
                        record["turn"] = 1
                    else:
                        record["turn"] = 0
                        
                    
                        
                    if self.is_val==True:
                        if self.use_low_time is True:
                            if int(record["white_remaining_time"])>30 or int(record["black_remaining_time"])>30:
                                continue
                        else:
                            if int(record["white_remaining_time"])<=30 or int(record["black_remaining_time"])<=30:
                                continue
                            if int(record["move_number"])<=8:
                                continue
                        
                    record["moves_until_end"] = record["moves_until_end"]//2 #number of full moves until the game ends
                            
                    is_continue = False
                    if self.min_full_move_number is not None:
                        if int(record["move_number"])//2 < self.min_full_move_number:
                            is_continue = True
                        else:
                            is_continue = False
                    if self.max_full_move_number is not None:
                        if int(record["move_number"])//2 > self.max_full_move_number:
                            is_continue = True
                        else:
                            is_continue = False
                    if is_continue is True:
                        continue
                    
                    yield {
                        "turn": torch.tensor([record["turn"]]).long(),
                        "white_kingside_castling_rights": torch.tensor([record["white_kingside_castling_rights"]]).long(),
                        "white_queenside_castling_rights": torch.tensor([record["white_queenside_castling_rights"]]).long(),
                        "black_kingside_castling_rights": torch.tensor([record["black_kingside_castling_rights"]]).long(),
                        "black_queenside_castling_rights": torch.tensor([record["black_queenside_castling_rights"]]).long(),
                        "board_position": torch.tensor(np.array(record["board_position"])).long(),
                        "from_squares": torch.tensor([record["from_square"]]).long(),
                        "to_squares": torch.tensor([record["to_square"]]).long(),
                        "lengths": torch.tensor([record["length"]]).long(),
                        "phase": torch.tensor([record["phase"]]).long(),
                        "game_result": torch.tensor([record["result"]]).long(),
                        "categorical_result": torch.tensor([record["categorical_result"]]).long(),
                        "time_control": time_control,
                        "white_remaining_time": torch.tensor([record["white_remaining_time"]]).long(),
                        "black_remaining_time": torch.tensor([record["black_remaining_time"]]).long(),
                        "white_rating": torch.tensor([record["white_rating"]]).long(),
                        "black_rating": torch.tensor([record["black_rating"]]).long(),
                        "move_time": torch.tensor([record["time_spent_on_move"]]).long(),
                        "move_number": torch.tensor([record["move_number"]]).long(),
                        "num_legal_moves": torch.tensor([record["num_legal_moves"]]).long(),
                        "white_material_value": torch.tensor([record["white_material_value"]]).long(),
                        "black_material_value": torch.tensor([record["black_material_value"]]).long(),
                        "material_difference": torch.tensor([record["material_difference"]]).long(),
                        "moves_until_end": torch.tensor([record["moves_until_end"]]).long(),
                        "base_time": torch.tensor([record["base_time"]]).long(),
                        "increment_time": torch.tensor([record["increment_time"]]).long(),
                    }


    def __len__(self):
        return self.length // self.world_size





if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name)
    
    
    # List of file paths to be processed.
    file_list = get_all_record_files('/Volumes/Lexar/1900_training_chunks')
    file_list = [file for file in file_list if file.endswith('.zst')]   
    #file_list = [file.replace("._", "", 1) for file in file_list]
    print(f"found {len(file_list)} chunks")
    
    # Instantiate the dataset with the list of files.
    dataset = ChunkLoader(file_list, record_dtype, 0, 0, False, False)
    
    # Create a DataLoader with multiple workers.
    loader = DataLoader(dataset, batch_size=CONFIG.BATCH_SIZE, num_workers=mp.cpu_count()//2)
    
    print("dataloader successfully created")
    
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
                
    dataiter = iter(cycle(loader))
    
    print("iterating dataset")
    
    # Iterate over the DataLoader.
    start = time.time()
    average = 0
    n = 0
    for batch in dataiter:
        # Process your batch here.
        elapsed = time.time()-start
        print(f"Time taken for one batch: {elapsed}s")
        start = time.time()
        
        average += elapsed
        n+=1
        print(f"Average time taken per batch: {average/n}s")