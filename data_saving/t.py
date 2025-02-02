import numpy as np
import torch
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

def get_all_record_files(data_folder):
        file_paths = []
        for folder in sorted(os.listdir(data_folder)):
            folder_path = os.path.join(data_folder, folder)
            if os.path.isdir(folder_path):
                for file in sorted(os.listdir(folder_path)):
                    if file.startswith("records_") and file.endswith(".zst"):
                        file_paths.append(os.path.join(folder_path, file))
        return file_paths



import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import numpy as np
import zstandard as zstd

class ZSTDIterableDataset(IterableDataset):
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

    def __iter__(self):
        # Get worker information for multi-worker support.
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
        global_idx = 0  # Global record counter across all files.
        # Iterate over each file.
        while True:
            for filename in self.file_list:
                # Ensure the file has the .zst extension.
                decompressor = zstd.ZstdDecompressor()
                with open(filename, 'rb') as f:
                    with decompressor.stream_reader(f) as zf:
                        while True:
                            record_bytes = zf.read(self.record_dtype.itemsize)
                            if not record_bytes:
                                break  # End of file
                            if global_idx % num_workers == worker_id:
                                record = np.frombuffer(record_bytes, dtype=self.record_dtype)
                                
                                curr_record = record[0]
                                try:
                                    base_time = curr_record[12]
                                    increment_time = curr_record[13]
                                    time_control = f'{base_time}+{increment_time}'
                                    time_control = torch.LongTensor([time_controls_encoded[time_control]])
                                except KeyError:
                                    time_control = torch.LongTensor([0])
                                yield {
                                    "turn": torch.tensor([curr_record[0]]),
                                    "white_kingside_castling_rights": torch.tensor([curr_record[1]]),
                                    "white_queenside_castling_rights": torch.tensor([curr_record[2]]),
                                    "black_kingside_castling_rights": torch.tensor([curr_record[3]]),
                                    "black_queenside_castling_rights": torch.tensor([curr_record[4]]),
                                    "board_position": torch.tensor(np.array(curr_record[5])),
                                    "from_square": torch.tensor([curr_record[6]]),
                                    "to_square": torch.tensor([curr_record[7]]),
                                    "length": torch.tensor([curr_record[8]]),
                                    "phase": torch.tensor([curr_record[9]]),
                                    "result": torch.tensor([curr_record[10]]),
                                    "categorical_result": torch.tensor([curr_record[11]]),
                                    "time_control": time_control,
                                    "white_remaining_time": torch.tensor([curr_record[14]]),
                                    "black_remaining_time": torch.tensor([curr_record[15]]),
                                    "white_rating": torch.tensor([curr_record[16]]),
                                    "black_rating": torch.tensor([curr_record[17]]),
                                    "time_spent_on_move": torch.tensor([curr_record[18]]),
                                    "move_number": torch.tensor([curr_record[19]]),
                                    "num_legal_moves": torch.tensor([curr_record[20]]),
                                    "white_material_value": torch.tensor([curr_record[21]]),
                                    "black_material_value": torch.tensor([curr_record[22]]),
                                    "material_difference": torch.tensor([curr_record[23]]),
                                    "moves_until_end": torch.tensor([curr_record[24]]),
                                }
                            global_idx += 1
                    

# Example usage:
if __name__ == "__main__":
    # List of file paths to be processed.
    file_list = get_all_record_files('data_folder')
    
    # Instantiate the dataset with the list of files.
    dataset = ZSTDIterableDataset(file_list, record_dtype)
    
    # Create a DataLoader with multiple workers.
    loader = DataLoader(dataset, batch_size=512, num_workers=4)
    
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
                
    dataiter = iter(cycle(loader))
    
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
