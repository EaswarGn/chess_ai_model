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




import multiprocessing as mp
from multiprocessing import Manager
from tqdm import tqdm
import os
import time
import numpy as np
import torch
import zstandard as zstd
import random
from torch.utils.data import Dataset
import sys

def f(str):
    print(str)
    sys.stdout.flush()


def fill_buffer_in_process(buffer, lock, start, data_file_paths, buffer_len, forever=False):
    pbar = tqdm(total=buffer_len, position=0, desc="Loading buffer")
    while True:
        with lock:  # Acquire lock for atomic check
            buffer_has_space = len(buffer) < buffer_len
        if buffer_has_space:
            #print("yes")
            # Get filename and increment start under lock
            with lock:
                filename = data_file_paths[start.value]
                start.value = (start.value + 1) % len(data_file_paths)
            # Process file and append records
            try:
                loaded_records = read_records_from_file(filename, record_dtype)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
            for record in loaded_records:
                curr_record = record[0]
                try:
                    base_time = curr_record[12]
                    increment_time = curr_record[13]
                    time_control = f'{base_time}+{increment_time}'
                    time_control = torch.LongTensor([time_controls_encoded[time_control]])
                except KeyError:
                    time_control = torch.LongTensor([0])

                record_dict = {
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
                with lock:
                    buffer.append(record_dict)
            pbar.update(len(loaded_records))
        else:
            if not forever:
                pbar.close()
                return
            time.sleep(0.01)  # Reduce CPU usage
    pbar.close()

def read_records_from_file(filename, dtype):
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

class ChunkParser:
    def __init__(self, data_folder=None, buffer_size=0, num_workers=4, **unused):
        #self.buffer = deque()
        manager = Manager()
        self.buffer = manager.list()
        self.buffer_len = buffer_size
        self.data_folder = data_folder
        self.start = manager.Value('i', 0)
        self.data_file_paths = self.get_all_record_files(data_folder)
        self.lock = manager.Lock()

        print("Filling initial buffer...")
        start_time = time.time()
        self.fill_buffer(forever=False)
        print(f"Took {time.time()-start_time}s to fill buffer")

        self.processes = []
        for _ in range(num_workers):
            process = mp.Process(target=fill_buffer_in_process, args=(self.buffer, self.lock, self.start, self.data_file_paths, self.buffer_len, True))
            #process = mp.Process(target=self.fill_buffer, args=(True, ))
            process.daemon = True
            process.start()
            self.processes.append(process)
            #process.join()
        print("Automatic buffer filling started with multiple processes.")

    def get_all_record_files(self, data_folder):
        file_paths = []
        for folder in sorted(os.listdir(data_folder)):
            folder_path = os.path.join(data_folder, folder)
            if os.path.isdir(folder_path):
                for file in sorted(os.listdir(folder_path)):
                    if file.startswith("records_") and file.endswith(".zst"):
                        file_paths.append(os.path.join(folder_path, file))
        return file_paths

    def fill_buffer(self, forever=False):
        pbar = tqdm(total=self.buffer_len, position=0, desc="Loading buffer")
        while True:
            with self.lock:  # Acquire lock for atomic check
                buffer_has_space = len(self.buffer) < self.buffer_len
            if buffer_has_space:
                # Get filename and increment start under lock
                with self.lock:
                    filename = self.data_file_paths[self.start.value]
                    self.start.value = (self.start.value + 1) % len(self.data_file_paths)
                # Process file and append records
                try:
                    loaded_records = read_records_from_file(filename, record_dtype)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    continue
                for record in loaded_records:
                    curr_record = record[0]
                    try:
                        base_time = curr_record[12]
                        increment_time = curr_record[13]
                        time_control = f'{base_time}+{increment_time}'
                        time_control = torch.LongTensor([time_controls_encoded[time_control]])
                    except KeyError:
                        time_control = torch.LongTensor([0])

                    record_dict = {
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
                    with self.lock:
                        self.buffer.append(record_dict)
                pbar.update(len(loaded_records))
            else:
                if not forever:
                    pbar.close()
                    return
                time.sleep(0.01) 
            
    def read_records_from_file(self, filename, dtype):
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

    def __getitem__(self, i):
        with self.lock:
            return self.buffer.pop(i)

    def __len__(self):
        with self.lock:
            return len(self.buffer)//4


    
    
if __name__ == '__main__':
    buffer_size = 200
    # Create ChunkParser instance (this will run fill_buffer() in the background)
    parser = ChunkParser(data_folder='data_folder', buffer_size=buffer_size, num_workers=6)

    train_loader = DataLoader(
        dataset=parser,
        batch_size=5,
        #num_workers=2,
        #pin_memory=False,
        #prefetch_factor=CONFIG.PREFETCH_FACTOR,
        shuffle=True,
    )
    
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
                
    dataiter = iter(cycle(train_loader))

    start = time.time()
    average = 0
    n=0
    for i, batch in enumerate(dataiter):
        elapsed = time.time()-start
        print("number of batches:",len(train_loader))
        print(f"Taken {elapsed}s to load batch")
        start = time.time()
        
        #print(f"buffer length: {len(parser.buffer)}")
        
        
        average += elapsed
        n+=1
        
        print(f"Average time taken: {average/n}s")
    
    # Continue with the rest of your script without waiting for the background task
    print("Main script continues running.")
    
