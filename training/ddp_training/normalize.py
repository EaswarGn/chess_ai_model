import math
import zstandard as zstd
import struct
from utils import get_all_record_files
import numpy as np
from tqdm import tqdm
import pickle
import sys


training_file_list = get_all_record_files('/Volumes/Lexar/1900_training_chunks')
training_file_list = [file for file in training_file_list if file.endswith('.zst')]   
training_file_list = [s for s in training_file_list if "._" not in s]
files = training_file_list


record_size = 109
fmt = "<5b64b6b2h2f2hf5hf"
                    

# Initialize a dictionary to store mean and std of specific features
stats = {
    "base_time": {"sum": 0, "sum_squared": 0, "count": 0},
    "increment_time": {"sum": 0, "sum_squared": 0, "count": 0},
    "white_remaining_time": {"sum": 0, "sum_squared": 0, "count": 0},
    "black_remaining_time": {"sum": 0, "sum_squared": 0, "count": 0},
    "white_rating": {"sum": 0, "sum_squared": 0, "count": 0},
    "black_rating": {"sum": 0, "sum_squared": 0, "count": 0},
    "time_spent_on_move": {"sum": 0, "sum_squared": 0, "count": 0},
    "move_number": {"sum": 0, "sum_squared": 0, "count": 0},
    "num_legal_moves": {"sum": 0, "sum_squared": 0, "count": 0},
    "white_material_value": {"sum": 0, "sum_squared": 0, "count": 0},
    "black_material_value": {"sum": 0, "sum_squared": 0, "count": 0},
    "material_difference": {"sum": 0, "sum_squared": 0, "count": 0},
    "moves_until_end": {"sum": 0, "sum_squared": 0, "count": 0},
}

# Function to calculate mean and std for a particular feature
def calculate_mean_std(stats):
    for key in stats:
        if stats[key]["count"] > 0:
            mean = stats[key]["sum"] / stats[key]["count"]
            variance = (stats[key]["sum_squared"] / stats[key]["count"]) - (mean ** 2)
            std_dev = math.sqrt(variance)
            stats[key]["mean"] = mean
            stats[key]["std"] = std_dev
      
total = 100000*len(files)
pbar = tqdm(total=total)

for filename in files:
    with open(filename, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        try:
            decompressed = dctx.decompress(f.read())
        except zstd.ZstdError as e:
            print(f"zstd error when reading {filename}")
            continue
        num_dicts = len(decompressed) // record_size
        
        
        for i in range(num_dicts):
            offset = i * record_size
            record_bytes = decompressed[offset: offset + record_size]
            unpacked = struct.unpack(fmt, record_bytes)
            record = {}
            idx = 0
            
            # Extracting values
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
            
            # Updating stats for each feature
            for feature in stats:
                value = record.get(feature)
                if value is not None:
                    stats[feature]["sum"] += value
                    stats[feature]["sum_squared"] += value ** 2
                    stats[feature]["count"] += 1
            pbar.update(1)

# After processing, calculate mean and std
calculate_mean_std(stats)

# Save the dictionary to a file using pickle
with open("stats_dict.pkl", "wb") as f:
    pickle.dump(stats, f)

# Print the result
for feature, values in stats.items():
    print(f"{feature}: Mean = {values.get('mean')}, Std = {values.get('std')}")
pbar.close()