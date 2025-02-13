import os
import json
import torch
import argparse
import tables as tb
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
import time
import zstandard as zstd
from zstandard import ZstdError
import struct

from configs import import_config
from time_controls import time_controls_encoded
import numpy as np


class ChessDatasetFT(Dataset):
    def __init__(self, data_folder, h5_file, split, **unused):
        """
        Init.

        Args:

            data_folder (str): The folder containing the H5 and splits
            files.

            h5_file (str): The H5 file.

            split (str): The data split. One of "train", "val", None.
            Defaults to None, which means that all datapoints will be
            included.
        """
        # Open table in H5 file
        self.h5_file = tb.open_file(os.path.join(data_folder, h5_file), mode="r")
        self.encoded_table = self.h5_file.root.encoded_data
        #self.human_table = self.h5_file.root.data
        self.len = self.encoded_table.nrows
        #self.split = split
        
        self.first_index = 0
        
    def convert_to_indices(self, result):
        if result == 1:  # white win
            return 2
        elif result == 0:  # draw
            return 1
        else:  # black win
            return 0

    def __getitem__(self, i):
        turns = torch.IntTensor([self.encoded_table[self.first_index + i]["turn"]])
        white_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["white_kingside_castling_rights"]]
        )  # (1)
        white_queenside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["white_queenside_castling_rights"]]
        )  # (1)
        black_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["black_kingside_castling_rights"]]
        )  # (1)
        black_queenside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["black_queenside_castling_rights"]]
        )  # (1)
        board_position = torch.IntTensor(
            self.encoded_table[self.first_index + i]["board_position"]
        )  # (64)
        from_square = torch.LongTensor(
            [self.encoded_table[self.first_index + i]["from_square"]]
        )  # (1)
        to_square = torch.LongTensor(
            [self.encoded_table[self.first_index + i]["to_square"]]
        )  # (1)
        length = torch.LongTensor([1])
        
       #new features
        phase = torch.IntTensor(
            [self.encoded_table[i]['phase']-2]
        )
        result = torch.IntTensor(
            [self.encoded_table[i]['result']]
        )
        
        categorical_result = torch.LongTensor(
            [self.convert_to_indices(self.encoded_table[i]['result'])]
        )
        
        try:
            base_time = self.encoded_table[i]['base_time']
            increment_time = self.encoded_table[i]['increment_time']
            time_control = f'{base_time}+{increment_time}'
            
            # Attempt to look up the time control encoding
            time_control = torch.LongTensor([time_controls_encoded[time_control]])

        except KeyError as e:
            # Handle the KeyError
            #print(f"KeyError: Missing key {e} in encoded_table or time_controls_encoded")
            # You can either provide a default value or skip the iteration, etc.
            base_time = None
            increment_time = None
            time_control = torch.LongTensor([0])  # Or whatever default you need

        
        """white_remaining_time = torch.FloatTensor(
            [self.encoded_table[i]['white_remaining_time']]
        )
        black_remaining_time = torch.FloatTensor(
            [self.encoded_table[i]['black_remaining_time']]
        )"""
        white_remaining_time = torch.FloatTensor(
            [self.encoded_table[i]['white_remaining_time']]
        )
        black_remaining_time = torch.FloatTensor(
            [self.encoded_table[i]['black_remaining_time']]
        )
        """white_rating = torch.IntTensor(
            [self.encoded_table[i]['white_rating']-1]
        )
        black_rating = torch.IntTensor(
            [self.encoded_table[i]['black_rating']-1]
        )"""
        white_rating = torch.IntTensor(
            [self.encoded_table[i]['white_rating']-1]
        )
        black_rating = torch.IntTensor(
            [self.encoded_table[i]['black_rating']-1]
        )
        """time_spent_on_move = torch.FloatTensor(
            [self.encoded_table[i]['time_spent_on_move']]
        )"""
        time_spent_on_move = torch.FloatTensor(
            [self.encoded_table[i]['time_spent_on_move']/100]
        )
        move_number = torch.IntTensor(
            [self.encoded_table[i]['move_number']]
        )
        num_legal_moves = torch.IntTensor(
            [self.encoded_table[i]['num_legal_moves']]
        )
        
        white_material_value = torch.IntTensor(
            [self.encoded_table[i]['white_material_value']]
        )
        
        black_material_value = torch.IntTensor(
            [self.encoded_table[i]['black_material_value']]
        )
        
        material_difference = torch.IntTensor(
            [self.encoded_table[i]['material_difference']]
        )
        
        moves_until_end = torch.FloatTensor(
            [self.encoded_table[i]['moves_until_end']/100]
        )
        
        return {
            "turns": turns,
            "white_kingside_castling_rights": white_kingside_castling_rights,
            "white_queenside_castling_rights": white_queenside_castling_rights,
            "black_kingside_castling_rights": black_kingside_castling_rights,
            "black_queenside_castling_rights": black_queenside_castling_rights,
            "board_positions": board_position,
            "from_squares": from_square,
            "to_squares": to_square,
            "lengths": length,
            "phase": phase,
            "game_result": result,
            "time_control": time_control,
            "white_remaining_time": white_remaining_time,
            "black_remaining_time": black_remaining_time,
            "white_rating": white_rating,
            "black_rating": black_rating,
            "move_time": time_spent_on_move,
            "move_number": move_number,
            "num_legal_moves": num_legal_moves,
            "white_material_value": white_material_value,
            "black_material_value": black_material_value,
            "material_difference": material_difference,
            "moves_until_end": moves_until_end,
            "categorical_result": categorical_result
        }

    def __len__(self):
        return self.len
    
    
from pathlib import Path
def get_all_record_files(directory: str):
    return [str(file) for file in Path(directory).rglob("*") if file.is_file()]





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
        self.length = len(self.file_list)*self.get_chunk_size()
        
    def get_chunk_size(self):
        with open(self.file_list[0], "rb") as f:
            dctx = zstd.ZstdDecompressor()
            
            decompressed = dctx.decompress(f.read())
            num_dicts = len(decompressed) // self.record_size
            return num_dicts

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
                    
                    if record["turn"]==0:
                        if record['white_remaining_time'] == 0:
                            record["time_spent_on_move"] = 0.05
                        else:
                            record["time_spent_on_move"] = record["time_spent_on_move"]/record["white_remaining_time"]
                    else:
                        if record['black_remaining_time'] == 0:
                            record["time_spent_on_move"] = 0.05
                        else:
                            record["time_spent_on_move"] = record["time_spent_on_move"]/record["black_remaining_time"]

                    # 5 int16:
                    record["move_number"] = unpacked[idx]; idx += 1
                    record["num_legal_moves"] = unpacked[idx]; idx += 1
                    record["white_material_value"] = unpacked[idx]; idx += 1
                    record["black_material_value"] = unpacked[idx]; idx += 1
                    record["material_difference"] = unpacked[idx]; idx += 1

                    # 1 float32:
                    record["moves_until_end"] = unpacked[idx]/100; idx += 1
                    
                    try:
                        base_time = record["base_time"]
                        increment_time = record["increment_time"]
                        time_control = f'{base_time}+{increment_time}'
                        time_control = torch.LongTensor([time_controls_encoded[time_control]])
                    except KeyError:
                        #time_control = torch.LongTensor([0])
                        continue

                    yield {
                        "turn": torch.tensor([record["turn"]]),
                        "white_kingside_castling_rights": torch.tensor([record["white_kingside_castling_rights"]]),
                        "white_queenside_castling_rights": torch.tensor([record["white_queenside_castling_rights"]]),
                        "black_kingside_castling_rights": torch.tensor([record["black_kingside_castling_rights"]]),
                        "black_queenside_castling_rights": torch.tensor([record["black_queenside_castling_rights"]]),
                        "board_position": torch.tensor(np.array(record["board_position"])),
                        "from_squares": torch.tensor([record["from_square"]]),
                        "to_squares": torch.tensor([record["to_square"]]),
                        "lengths": torch.tensor([record["length"]]),
                        "phase": torch.tensor([record["phase"]]),
                        "game_result": torch.tensor([record["result"]]),
                        "categorical_result": torch.tensor([record["categorical_result"]]),
                        "time_control": time_control,
                        "white_remaining_time": torch.tensor([record["white_remaining_time"]]),
                        "black_remaining_time": torch.tensor([record["black_remaining_time"]]),
                        "white_rating": torch.tensor([record["white_rating"]]),
                        "black_rating": torch.tensor([record["black_rating"]]),
                        "move_time": torch.tensor([record["time_spent_on_move"]]),
                        "move_number": torch.tensor([record["move_number"]]),
                        "num_legal_moves": torch.tensor([record["num_legal_moves"]]),
                        "white_material_value": torch.tensor([record["white_material_value"]]),
                        "black_material_value": torch.tensor([record["black_material_value"]]),
                        "material_difference": torch.tensor([record["material_difference"]]),
                        "moves_until_end": torch.tensor([record["moves_until_end"]]),
                    }
    def __len__(self): 
        return self.length




if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name)

    # Dataset
    dataset = ChessDatasetFT(
        data_folder='',
        h5_file='data.h5',
        split="train",
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=512,
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
        print(f"Taken {elapsed}s to load batch")
        start = time.time()
    dataset.h5_file.close()