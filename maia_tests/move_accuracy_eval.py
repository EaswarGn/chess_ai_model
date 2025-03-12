from maia2 import model, dataset, inference
from tqdm import tqdm
from pathlib import Path
from zstandard import ZstdError
import zstandard as zstd
import struct
from levels import SQUARES
import sys
from utils import *
import chess
from configs import import_config
import time

def get_all_record_files(directory: str):
    return [str(file) for file in Path(directory).rglob("*") if file.is_file()]

def run_tests(harmonia_model):
    maia2_model = model.from_pretrained(type="blitz", device="cpu")
    data = dataset.load_example_test_dataset()
    prepared = inference.prepare()
    maia_correct = 0
    file_list = get_all_record_files('/Volumes/Lexar/1_chunks')
    file_list = [file for file in file_list if file.endswith('.zst')]   
    file_list = [s for s in file_list if "._" not in s]

    total = (len(file_list)*100000)//5000
    pbar = tqdm(total=total)
    
    if len(file_list)==0:
        print("no files to evaluate with.")
        return
    
    incrementer = 0

    harmonia_correct = 0
    record_size = 309
    fmt = "<5b64b6b2h2f2hf5hf200s"
    
    start_time = time.time()
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
                    if char!='\\':
                        new_fen_string += char
                    if char=='\\':
                        break
                fen = new_fen_string[2:].strip()
                _, turn, castling_rights, ep_square, halfmove_count, fullmove_count = fen.split()
                """print("fen full move number: ",fullmove_count)
                print("record full move number: ",record["move_number"])
                print()"""
                
                #skip moves made under low time
                if int(record["white_remaining_time"]) < 30 or int(record["black_remaining_time"]) < 30:
                    continue
                
                elo_self = 0
                elo_oppo = 0
                if int(record['turn']) == 0:
                    elo_self = int(record["black_rating"])
                    elo_oppo = int(record["white_rating"])
                else:
                    elo_oppo = int(record["black_rating"])
                    elo_self = int(record["white_rating"])
                
                reversed_squares = {v: k for k, v in SQUARES.items()}
                move = reversed_squares[record["from_square"]] + reversed_squares[record["to_square"]]
                move_probs, win_prob = inference.inference_each(maia2_model, prepared, fen, elo_self, elo_oppo)
                
                if max(move_probs, key=move_probs.get) == move:
                    maia_correct += 1
                    
                board = chess.Board(fen)
                predictions = harmonia_model(get_model_inputs(board,
                                            time_control=f"{record['base_time']}+{record['increment_time']}",
                                            white_remaining_time=record["white_remaining_time"],
                                            black_remaining_time=record["black_remaining_time"],
                                            white_rating=record["white_rating"],
                                            black_rating=record["black_rating"])
                            )
                model_move = get_move(board, predictions)
                
                if model_move == move:
                    harmonia_correct += 1

                    
                incrementer += 1
            
                incrementer += 1
                pbar.n = incrementer
                pbar.refresh()
                if pbar.n >= total:
                    pbar.close()
                    print(f"Maia accuracy on test set: {round(maia_correct/total, 4)}")
                    print(f"Harmonia accuracy on test set: {round(harmonia_correct/total, 4)}")
                    sys.exit()


if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name)
    CONFIG = CONFIG.CONFIG()

    # Train model
    harmonia_model = load_model(CONFIG)
    run_tests(harmonia_model)
    