from maia2 import model, dataset, inference
from tqdm import tqdm
from pathlib import Path
from zstandard import ZstdError
import zstandard as zstd
import struct
from levels import SQUARES
import sys

def get_all_record_files(directory: str):
    return [str(file) for file in Path(directory).rglob("*") if file.is_file()]

maia2_model = model.from_pretrained(type="blitz", device="cpu")
data = dataset.load_example_test_dataset()
prepared = inference.prepare()
correct = 0

file_list = get_all_record_files('/Volumes/Lexar/1_chunks')
file_list = [file for file in file_list if file.endswith('.zst')]   
file_list = [s for s in file_list if "._" not in s]

total = (len(file_list)*100000)//5000
pbar = tqdm(total=total)


record_size = 309
fmt = "<5b64b6b2h2f2hf5hf200s"
for filename in file_list:
    with open(filename, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        
        decompressed = dctx.decompress(f.read())
        num_dicts = len(decompressed) // record_size
        for i in range(num_dicts):
            offset = i * record_size
            record_bytes = decompressed[offset: offset + record_size]
            unpacked = struct.unpack(fmt, record_bytes)
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
            
            # fen string (200 bytes)
            record["fen"] = unpacked[idx:idx+200]; idx += 1
            fen_string = str(record["fen"][0])
            new_fen_string = ''
            for char in fen_string:
                if char!='x' and char!='0' and char!='\\':
                    new_fen_string += char
            fen = new_fen_string[2:len(new_fen_string)-2]
            
            elo_self = 0
            elo_oppo = 0
            if int(record['turn']) == "0":
                elo_self = int(record["black_rating"])
                elo_oppo = int(record["white_rating"])
            else:
                elo_oppo = int(record["black_rating"])
                elo_self = int(record["white_rating"])
            
            reversed_squares = {v: k for k, v in SQUARES.items()}
            move = reversed_squares[record["from_square"]] + reversed_squares[record["to_square"]]
            move_probs, win_prob = inference.inference_each(maia2_model, prepared, fen, elo_self, elo_oppo)
            
            if max(move_probs, key=move_probs.get) == move:
                correct += 1
                
            pbar.update(1)
            if pbar.n >= total:
                print(correct/total)
                sys.exit()

