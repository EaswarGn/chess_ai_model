from pathlib import Path
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
import chess
from tqdm import tqdm
import math

from configs import import_config
from time_controls import time_controls_encoded
import numpy as np
import multiprocessing as mp
import pickle
from configs.models.utils.levels import *


def get_all_record_files(directory: str):
    return [str(file) for file in Path(directory).rglob("*") if file.is_file()]

file_list = get_all_record_files('/Volumes/Lexar/1900_training_chunks')
file_list = [file for file in file_list if file.endswith('.zst')]   
file_list = [s for s in file_list if "._" not in s]
print(f"found {len(file_list)} chunks")

white_win_count = 0
draw_count = 0
black_win_count = 0

total = len(file_list) * 100000
pbar = tqdm(total=total, dynamic_ncols=True)

record_size = 109
fmt = "<5b64b6b2h2f2hf5hf"
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
            
            board = chess.Board(None)
            reversed_squares = {value: key for key, value in SQUARES.items()}
            
            board.fullmove_number = record["move_number"]
            if record['turn'] == 1:
                board.turn = chess.BLACK
            if record['turn'] == 0:
                board.turn = chess.WHITE
                
            for index, piece in enumerate(record["board_position"]):
                if piece != 0:
                    square_str = reversed_squares[index]
                    square = chess.SQUARE_NAMES.index(square_str)
                    
                    curr_piece = None
                    if piece==1:
                        board.ep_square = square
                    if piece==2:
                        curr_piece = chess.Piece(chess.PAWN, chess.WHITE)
                    if piece==3:
                        curr_piece = chess.Piece(chess.PAWN, chess.BLACK)
                    if piece==4:
                        curr_piece = chess.Piece(chess.ROOK, chess.WHITE)
                    if piece==5:
                        curr_piece = chess.Piece(chess.ROOK, chess.BLACK)
                    if piece==6:
                        curr_piece = chess.Piece(chess.KNIGHT, chess.WHITE)
                    if piece==7:
                        curr_piece = chess.Piece(chess.KNIGHT, chess.BLACK)
                    if piece==8:
                        curr_piece = chess.Piece(chess.BISHOP, chess.WHITE)
                    if piece==9:
                        curr_piece = chess.Piece(chess.BISHOP, chess.BLACK)
                    if piece==10:
                        curr_piece = chess.Piece(chess.QUEEN, chess.WHITE)
                    if piece==11:
                        curr_piece = chess.Piece(chess.QUEEN, chess.BLACK)
                    if piece==12:
                        curr_piece = chess.Piece(chess.KING, chess.WHITE)
                    if piece==13:
                        curr_piece = chess.Piece(chess.KING, chess.BLACK)

                    board.set_piece_at(square, curr_piece)

            num_legal_moves = len(list(board.legal_moves))
            
            if abs(record["num_legal_moves"] - num_legal_moves) > 1:
                print(record)
                print(board.fen())
                print(num_legal_moves)
                sys.exit()
            #sys.exit()
                    
                    
            
            if int(record['move_number']) <= 8:
                record["moves_until_end"] = 35

            try:
                base_time = record["base_time"]
                increment_time = record["increment_time"]
                
                if int(increment_time) > 0 and int(record["move_number"]) != 0:
                    record["time_spent_on_move"] = record["time_spent_on_move"] + int(increment_time)
                if int(record["time_spent_on_move"]) < 1:
                    record["time_spent_on_move"] = 0
                
                time_control = f'{base_time}+{increment_time}'
                time_control = torch.FloatTensor([5.0])
            except KeyError:
                pass

            if int(record["turn"]) == 0:
                record["turn"] = 1
            else:
                record["turn"] = 0
                
            result = int(record["categorical_result"])
            if result == 2:
                white_win_count += 1
            elif result == 1:
                draw_count += 1
            elif result == 0:
                black_win_count += 1

            # Update the progress bar
            pbar.update(1)
            if i % 1000 == 0:  # Update postfix every 1000 iterations to reduce overhead
                pbar.set_postfix({
                    "White Wins": f"{white_win_count}({round(white_win_count/pbar.n, 3)})",
                    "Draws": f"{draw_count}({round(draw_count/pbar.n, 3)})",
                    "Black Wins": f"{black_win_count}({round(black_win_count/pbar.n, 3)})"
                })

print(f"total number of white wins: {white_win_count}({round(white_win_count/pbar.n, 3)})")
print(f"total number of draws: {draw_count}({round(draw_count/pbar.n, 3)})")
print(f"total number of black wins: {black_win_count}({round(black_win_count/pbar.n, 3)})")
pbar.close()