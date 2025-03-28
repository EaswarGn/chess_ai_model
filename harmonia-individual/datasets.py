import os
import json
import torch
import argparse
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import time
import zstandard as zstd
import torch.distributed as dist
import struct
import random
import sys

from configs import import_config
import numpy as np
import multiprocessing as mp
import pickle
import numpy as np

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
                 include_low_time_moves=False,
                 min_full_move_number= -1, #initialized to arbitrary small value to allow any minimum move number
                 max_full_move_number= 500, #initialized to arbitrary large value to allow any max move number
                 target_player=None,
                 loop_forever=False
        ):
        
        assert target_player is not None
        
        self.file_list = file_list
        self.record_dtype = record_dtype
        self.record_size = record_dtype.itemsize
        self.fmt = "<5b64b6b2h2f2hf5hf200s100s100s"
        self.record_size = 109+200+100+100  #struct.calcsize(self.fmt)
        self.length = len(self.file_list) * self.get_chunk_size()

        # Get rank and world size for distributed training
        self.rank = rank
        self.world_size = world_size
        self.include_low_time_moves = include_low_time_moves
        self.min_full_move_number = min_full_move_number
        self.max_full_move_number = max_full_move_number
        self.target_player = target_player
        self.loop_forever = loop_forever

    def get_chunk_size(self):
        with open(self.file_list[0], "rb") as f:
            dctx = zstd.ZstdDecompressor()
            decompressed = dctx.decompress(f.read())
            num_dicts = len(decompressed) // self.record_size
            return num_dicts
        
    def convert_byte_str(self, string):
        return string.decode('utf-8', errors='ignore')
        

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Shard dataset across GPUs
        files = self.file_list
        if self.world_size != 0:
            files = self.file_list[self.rank::self.world_size]

        # Further shard among DataLoader workers
        files = files[worker_id::num_workers]

        while True:
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
                        
                        # fen string (200 bytes)
                        record["fen"] = unpacked[idx:idx+200]; idx += 1
                        fen_string = self.convert_byte_str(record["fen"][0]).rstrip('\x00')
                        
                        record["white_player"] = unpacked[idx:idx+100]; idx += 1
                        white_player = self.convert_byte_str(record["white_player"][0]).rstrip('\x00')
                        record["black_player"] = unpacked[idx:idx+100]; idx += 1
                        black_player = self.convert_byte_str(record["black_player"][0]).rstrip('\x00')
                        
                        if int(record["turn"]) == 0:
                            record["turn"] = 1
                        else:
                            record["turn"] = 0


                        if record['turn'] == 1:
                            if white_player == self.target_player:
                                pass
                            else:
                                continue
                        if record['turn'] == 0:
                            if black_player == self.target_player:
                                pass
                            else:
                                continue

                        
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
                            time_control = torch.FloatTensor([5.0])
                        except KeyError:
                            pass
                            
                        if self.include_low_time_moves is True:
                            pass
                        else:
                            if int(record["white_remaining_time"])<=30 or int(record["black_remaining_time"])<=30:
                                continue
                                
                                
                        if int(record["move_number"])>self.min_full_move_number and int(record["move_number"])<self.max_full_move_number:
                            pass
                        else:
                            continue
                        
                        
                        #normalizing through log transformations
                        #record['time_spent_on_move'] = np.log1p(record['time_spent_on_move'])
                        
                        yield {
                            "turn": torch.tensor([record["turn"]]).float(), #make float
                            "white_kingside_castling_rights": torch.tensor([record["white_kingside_castling_rights"]]).float(),
                            "white_queenside_castling_rights": torch.tensor([record["white_queenside_castling_rights"]]).float(),
                            "black_kingside_castling_rights": torch.tensor([record["black_kingside_castling_rights"]]).float(),
                            "black_queenside_castling_rights": torch.tensor([record["black_queenside_castling_rights"]]).float(),
                            "board_position": torch.tensor(np.array(record["board_position"])).float(),
                            "from_squares": torch.tensor([record["from_square"]]).float(),
                            "to_squares": torch.tensor([record["to_square"]]).float(),
                            "lengths": torch.tensor([record["length"]]).float(),
                            "phase": torch.tensor([record["phase"]]).float(),
                            "game_result": torch.tensor([record["result"]]).float(),
                            "categorical_result": torch.tensor([record["categorical_result"]]).float(),
                            "time_control": time_control,
                            "white_remaining_time": torch.tensor([record["white_remaining_time"]]).float(),
                            "black_remaining_time": torch.tensor([record["black_remaining_time"]]).float(),
                            "white_rating": torch.tensor([record["white_rating"]]).float(),
                            "black_rating": torch.tensor([record["black_rating"]]).float(),
                            "move_time": torch.tensor([record["time_spent_on_move"]]).float(),
                            "move_number": torch.tensor([record["move_number"]]).float(),
                            "num_legal_moves": torch.tensor([record["num_legal_moves"]]).float(),
                            "white_material_value": torch.tensor([record["white_material_value"]]).float(),
                            "black_material_value": torch.tensor([record["black_material_value"]]).float(),
                            "material_difference": torch.tensor([record["material_difference"]]).float(),
                            "moves_until_end": torch.tensor([record["moves_until_end"]]).float(),
                            "base_time": torch.tensor([record["base_time"]]).float(),
                            "increment_time": torch.tensor([record["increment_time"]]).float(),
                        }
            if not self.loop_forever:
                break


    def __len__(self):
        return self.length // self.world_size





if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name).CONFIG()
    
    
    # List of file paths to be processed.
    file_list = get_all_record_files('data-prep/Chess_Star1234_data')
    file_list = [file for file in file_list if file.endswith('.zst')]   
    #file_list = [file.replace("._", "", 1) for file in file_list]
    print(f"found {len(file_list)} chunks")
    
    # Instantiate the dataset with the list of files.
    dataset = ChunkLoader(file_list, record_dtype, 0, 0, False, False, target_player='BlitzKing45')
    
    # Create a DataLoader with multiple workers.
    loader = DataLoader(dataset, batch_size=CONFIG.BATCH_SIZE, num_workers=1)#mp.cpu_count()//2)
    
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