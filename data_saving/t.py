import numpy as np
import torch
import struct
import time
from torch.utils.data import Dataset, DataLoader
import asyncio
import os
import random
from time_controls import time_controls_encoded
import zstandard as zstd
import sys
import threading
from tqdm import tqdm
from collections import deque

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
"""
# Create a structured numpy array with the specified dtype
def create_sample_record():
    return np.array(
        (
            1,  # turn
            1,  # white_kingside_castling_rights
            0,  # white_queenside_castling_rights
            1,  # black_kingside_castling_rights
            0,  # black_queenside_castling_rights
            np.random.randint(0, 6, size=(64,), dtype=np.int8),  # board_position
            10,  # from_square
            20,  # to_square
            1, # length
            2,  # phase
            1,  # result
            2, # categorical_result
            210, #base_time
            3, #increment_time
            np.float16(150.5),  # white_remaining_time
            np.float16(120.0),  # black_remaining_time
            1600,  # white_rating
            1550,  # black_rating
            np.float16(5.5),  # time_spent_on_move
            15,  # move_number
            20,  # num_legal_moves
            40,  # white_material_value
            35,  # black_material_value
            5,  # material_difference
            np.float16(10.0)  # moves_until_end
        ),
        dtype=record_dtype
    )

# Create multiple records
records = [create_sample_record() for _ in range(100)]

# Save the records to a compressed .zst file
def save_records_to_file(records, filename):
    filename += ".zst"  # Ensure .zst extension
    compressor = zstd.ZstdCompressor()
    with open(filename, 'wb') as f:
        with compressor.stream_writer(f) as zf:
            for record in records:
                zf.write(record.tobytes())  # Convert structured array to bytes and write

# Read the records from the .zst file
def read_records_from_file(filename, dtype):
    filename += ".zst"  # Ensure .zst extension
    records = []
    decompressor = zstd.ZstdDecompressor()
    with open(filename, 'rb') as f:
        with decompressor.stream_reader(f) as zf:
            while True:
                record_bytes = zf.read(dtype.itemsize)
                if not record_bytes:
                    break  # End of file
                record = np.frombuffer(record_bytes, dtype=dtype)
                records.append(record)
    return records

# Example usage
start = time.time()
save_records_to_file(records, "records")
print(f"Time taken to save records: {time.time()-start}s")

start = time.time()
loaded_records = read_records_from_file("records", record_dtype)
print(f"Time taken to read records: {time.time()-start}s")
print(len(loaded_records))

# Convert records to PyTorch tensors
start = time.time()
torch_tensors = [torch.tensor(record) for record in loaded_records[0][0]]
print(f"Time taken to create tensors: {time.time()-start}s")"""

from pathlib import Path
def get_all_record_files(directory: str):
    return [str(file) for file in Path(directory).rglob("*") if file.is_file()]



import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import numpy as np
import zstandard as zstd
from zstandard import ZstdError

class ChunkLoader(IterableDataset):
    """
    An IterableDataset that streams records from a list of Zstandard-compressed
    binary files. It iterates over each file in the provided list, reads fixed-size
    records from the decompressed stream, converts them into PyTorch tensors, and
    partitions the work across multiple DataLoader workers.
    """
    def __init__(self, file_list, record_dtype):
        """
        Args:
            file_list (list of str): List of paths to .zst files. If a file does not
                                     end with '.zst', the extension will be appended.
            record_dtype (np.dtype): Numpy structured dtype defining one record.
        """
        self.file_list = file_list
        self.record_dtype = record_dtype
        self.record_size = record_dtype.itemsize
        self.fmt = "<5b64b6b2h2f2hf5hf"
        self.record_size = struct.calcsize(self.fmt)

    def __iter__(self):
        # Get worker information for multi-worker support.
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
        #divide chunks amongst workers
        files = self.file_list[worker_id::num_workers]
        
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

                    # board_position: next 64 int8 values
                    record["board_position"] = list(unpacked[idx: idx+64]); idx += 64

                    # Next 6 int8 values:
                    record["from_square"] = unpacked[idx]; idx += 1
                    record["to_square"] = unpacked[idx]; idx += 1
                    record["length"] = unpacked[idx]; idx += 1
                    record["phase"] = unpacked[idx]; idx += 1
                    record["result"] = unpacked[idx]; idx += 1
                    record["categorical_result"] = unpacked[idx]; idx += 1

                    # 2 int16:
                    record["base_time"] = unpacked[idx]; idx += 1
                    record["increment_time"] = unpacked[idx]; idx += 1

                    # 2 float32:
                    record["white_remaining_time"] = unpacked[idx]; idx += 1
                    record["black_remaining_time"] = unpacked[idx]; idx += 1

                    # 2 int16:
                    record["white_rating"] = unpacked[idx]; idx += 1
                    record["black_rating"] = unpacked[idx]; idx += 1

                    # 1 float32:
                    record["time_spent_on_move"] = unpacked[idx]; idx += 1

                    # 5 int16:
                    record["move_number"] = unpacked[idx]; idx += 1
                    record["num_legal_moves"] = unpacked[idx]; idx += 1
                    record["white_material_value"] = unpacked[idx]; idx += 1
                    record["black_material_value"] = unpacked[idx]; idx += 1
                    record["material_difference"] = unpacked[idx]; idx += 1

                    # 1 float32:
                    record["moves_until_end"] = unpacked[idx]; idx += 1
                    
                    try:
                        base_time = record["base_time"]
                        increment_time = record["increment_time"]
                        time_control = f'{base_time}+{increment_time}'
                        time_control = torch.LongTensor([time_controls_encoded[time_control]])
                    except KeyError:
                        time_control = torch.LongTensor([0])

                    yield {
                        "turn": torch.tensor([record["turn"]]),
                        "white_kingside_castling_rights": torch.tensor([record["white_kingside_castling_rights"]]),
                        "white_queenside_castling_rights": torch.tensor([record["white_queenside_castling_rights"]]),
                        "black_kingside_castling_rights": torch.tensor([record["black_kingside_castling_rights"]]),
                        "black_queenside_castling_rights": torch.tensor([record["black_queenside_castling_rights"]]),
                        "board_position": torch.tensor(np.array(record["board_position"])),
                        "from_square": torch.tensor([record["from_square"]]),
                        "to_square": torch.tensor([record["to_square"]]),
                        "length": torch.tensor([record["length"]]),
                        "phase": torch.tensor([record["phase"]]),
                        "result": torch.tensor([record["result"]]),
                        "categorical_result": torch.tensor([record["categorical_result"]]),
                        "time_control": time_control,
                        "white_remaining_time": torch.tensor([record["white_remaining_time"]]),
                        "black_remaining_time": torch.tensor([record["black_remaining_time"]]),
                        "white_rating": torch.tensor([record["white_rating"]]),
                        "black_rating": torch.tensor([record["black_rating"]]),
                        "time_spent_on_move": torch.tensor([record["time_spent_on_move"]]),
                        "move_number": torch.tensor([record["move_number"]]),
                        "num_legal_moves": torch.tensor([record["num_legal_moves"]]),
                        "white_material_value": torch.tensor([record["white_material_value"]]),
                        "black_material_value": torch.tensor([record["black_material_value"]]),
                        "material_difference": torch.tensor([record["material_difference"]]),
                        "moves_until_end": torch.tensor([record["moves_until_end"]]),
                    }
                    

# Example usage:
if __name__ == "__main__":
    # List of file paths to be processed.
    file_list = get_all_record_files('/Volumes/Lexar/chessmodel_dataset/1900_training_chunks')
    file_list = [file for file in file_list if file.endswith('.zst')]   
    #file_list = [file.replace("._", "", 1) for file in file_list]
    print(len(file_list))
    
    # Instantiate the dataset with the list of files.
    dataset = ChunkLoader(file_list, record_dtype)
    
    # Create a DataLoader with multiple workers.
    loader = DataLoader(dataset, batch_size=512, num_workers=4)
    
    print("dataset created")
    
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
