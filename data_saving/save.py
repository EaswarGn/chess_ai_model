import tensorflow as tf
import chess
import chess.pgn
import os
import h5py
import tables as tb
import multiprocessing as mp
import time
from urllib3.exceptions import ProtocolError
import requests
import zstandard as zstd
import io
import threading
from tools import TURN, PIECES, SQUARES, UCI_MOVES, BOOL
from data_utils import is_increment, calculate_material, encode, get_move_numbers, is_increment_greater_than_base, time_to_seconds, parse_fen, extract_clock_times_from_pgn, get_piece_stats, print_progress_bar, get_file_size_in_mb, is_berserk_game
import yaml
import sys
import argparse
import random
import re

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
DATA_FOlDER = 'data'
max_move_sequence_length = cfg['model']['n_moves']
num_epochs = 3
curriculum = {
    'epoch_1': {
        'opening': 0.5,
        'middlegame': 0.3,
        'endgame': 0.2,
    },
    'epoch_2': {
        'opening': 0.3,
        'middlegame': 0.5,
        'endgame': 0.2,
    },
    'epoch_3': {
        'opening': 0.2,
        'middlegame': 0.5,
        'endgame': 0.3,
    }
}

phase_encoder = {
    'opening': 2,
    'middlegame': 3,
    'endgame': 4
}



# Create table description for H5 file
class ChessTable(tb.IsDescription):
    board_position = tb.StringCol(64)
    #raw_fen = tb.StringCol(80)
    turn = tb.StringCol(1)
    white_kingside_castling_rights = tb.BoolCol()
    white_queenside_castling_rights = tb.BoolCol()
    black_kingside_castling_rights = tb.BoolCol()
    black_queenside_castling_rights = tb.BoolCol()
    moves = tb.StringCol(
        shape=(max_move_sequence_length + 1), itemsize=8, dflt="<pad>"
    )  # "dflt" doesn't work for some reason
    length = tb.Int8Col()
    from_square = tb.StringCol(2)
    to_square = tb.StringCol(2)
    phase = tb.Int8Col()
    result = tb.StringCol(3)
    base_time = tb.Int16Col()
    increment_time = tb.Int16Col()
    white_remaining_time = tb.Int16Col()
    black_remaining_time = tb.Int16Col()
    white_rating = tb.Int16Col()
    black_rating = tb.Int16Col()
    time_spent_on_move = tb.Int16Col()
    move_number = tb.Int16Col()
    num_legal_moves = tb.Int16Col()
    white_material_value = tb.Int16Col()
    black_material_value = tb.Int16Col()
    material_difference = tb.Int16Col()
    moves_until_end = tb.Int16Col()

# Create table description for HDF5 file
class EncodedChessTable(tb.IsDescription):
    board_position = tb.Int8Col(shape=(64))
    turn = tb.Int8Col()
    white_kingside_castling_rights = tb.Int8Col()
    white_queenside_castling_rights = tb.Int8Col()
    black_kingside_castling_rights = tb.Int8Col()
    black_queenside_castling_rights = tb.Int8Col()
    moves = tb.Int16Col(shape=(max_move_sequence_length + 1))
    length = tb.Int8Col()
    from_square = tb.Int8Col()
    to_square = tb.Int8Col()
    phase = tb.Int8Col()
    result = tb.Int8Col()
    base_time = tb.Int16Col()
    increment_time = tb.Int16Col()
    white_remaining_time = tb.Float32Col()
    black_remaining_time = tb.Float32Col()
    white_rating = tb.Float32Col()
    black_rating = tb.Float32Col()
    time_spent_on_move = tb.Float32Col()
    move_number = tb.Int16Col()
    num_legal_moves = tb.Int16Col()
    white_material_value = tb.Int16Col()
    black_material_value = tb.Int16Col()
    material_difference = tb.Int16Col()
    moves_until_end = tb.Int16Col()
    



def write_to_h5(
    data_folder,
    epoch,
    exp_rows,
    type,
    url,
    min_rating=1600,
    max_rating=1700,
    h5_file='data.h5'
):
    # Delete H5 file if it already exists; start anew
    if os.path.exists(os.path.join(data_folder, h5_file)):
        os.remove(os.path.join(data_folder, h5_file))

    # Create new H5 file
    h5_file = tb.open_file(
        os.path.join(data_folder, h5_file), mode="w", title=f"{type} data file"
    )
    
    # Create table in H5 file
    table = h5_file.create_table("/", "data", ChessTable, expectedrows=exp_rows)

    # Create encoded table in H5 file
    encoded_table = h5_file.create_table(
        "/", "encoded_data", EncodedChessTable, expectedrows=table.nrows
    )
    
    row = table.row
    encoded_row = encoded_table.row
    last_processed_game_index = 0
    
    opening_count = 0
    middlegame_count = 0
    endgame_count = 0
    
    max_openings = int(exp_rows*curriculum[f'epoch_{epoch+1}']['opening'])
    max_middlegames = int(exp_rows*curriculum[f'epoch_{epoch+1}']['middlegame'])
    max_endgames = int(exp_rows*curriculum[f'epoch_{epoch+1}']['endgame'])
    
    while True:
        try:
            start_time = time.time()
            with requests.Session() as session:
                response = session.get(url, stream=True)
                response.raise_for_status()

                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(response.raw) as stream_reader:
                    decompressed_file = io.TextIOWrapper(stream_reader, encoding='utf-8', errors='replace')

                    # Skip to the last processed game
                    start = time.time()
                    for _ in range(last_processed_game_index):
                        m = chess.pgn.read_game(decompressed_file)
                    elapsed = time.time() - start
                    if last_processed_game_index > 0:
                        print(f"time taken to skip {last_processed_game_index} games: {elapsed:.4f}s")

                    game = chess.pgn.read_game(decompressed_file)
                    start_data_time = time.time()
                    while game:
                        
                        #loop to find a valid game
                        while (game.headers.get("Event") == 'Rated Correspondence game' or
                                game.headers.get("WhiteElo") == '?' or 
                                game.headers.get("BlackElo") == '?' or
                                game.headers.get("TimeControl") == '-' or
                                game.headers.get("Termination") == 'Abandoned' or
                                is_berserk_game(game) or
                                #is_increment_greater_than_base(game) or
                                is_increment(game)
                            ):
                            last_processed_game_index += 1  # Update the index for the next game
                            game = chess.pgn.read_game(decompressed_file)
                        
                        white_rating = game.headers.get("WhiteElo")
                        black_rating = game.headers.get("BlackElo")
                        result = game.headers.get("Result")
                        time_control = game.headers.get("TimeControl")
                        eco = game.headers.get("ECO")
                        opening = game.headers.get("Opening")
                        utc_time = game.headers.get("UTCTime")
                        termination = game.headers.get("Termination")
                        
                        white_rating = int(white_rating)
                        black_rating = int(black_rating)
                        
                        
                        base_time, increment_time = map(int, time_control.split("+"))
                        
                        
                        
                        # Extracting moves
                        moves = []
                        node = game
                        while node.variations:
                            move = node.variations[0].move
                            moves.append(move)
                            node = node.variations[0]

                        num_moves = len(moves)
                        
                        if(
                            (white_rating >= min_rating and white_rating<=max_rating)
                            and (black_rating >= min_rating and black_rating<=max_rating)
                            and (num_moves >= max_move_sequence_length*3)
                        ):
                            board = chess.Board()
                            fens = []
                            fens.append(board.fen())
                            moves = []
                            node = game
                            while node.variations:
                                move = node.variations[0].move
                                moves.append(move.uci())
                                node = node.variations[0]
                                board.push(move)
                                fens.append(board.fen())
                            white_clock_times, black_clock_times, white_elapsed_times, black_elapsed_times = extract_clock_times_from_pgn(game)
                            move_numbers = get_move_numbers(game)
                            
                            #start_index = 0 if result == "1-0" else 1
                            start_index = 0
                            
                            if opening_count == max_openings:
                                start_index += 10
                                if middlegame_count == max_middlegames:
                                    start_index+=10
                            
                            
                            for k in range(start_index, len(moves)):
                                if exp_rows == opening_count+middlegame_count+endgame_count:
                                    h5_file.close()
                                    datapoints = opening_count+middlegame_count+endgame_count
                                    print(f"{datapoints} datapoints saved to disk. {last_processed_game_index} games processed.")
                                    return
                                
                                #directory = f'data/epoch_{epoch+1}/train_data'
                                
                                current = [opening_count+middlegame_count+endgame_count,
                                           opening_count,
                                           middlegame_count,
                                           endgame_count]
                                total = [exp_rows, max_openings, max_middlegames, max_endgames]
                                names = ['total', 'openings', 'middlegames', 'endgames']
                                
                                progress_message = ""
                                for i in range(4):
                                    bar_length = 40
                                    percentage = (current[i] / total[i]) * 100
                                    # Calculate the number of dashes to represent the progress
                                    num_dashes = int(bar_length * (current[i] / total[i]))
                                    
                                    # Create the progress bar string with dashes
                                    bar = '-' * num_dashes + ' ' * (bar_length - num_dashes)
                                    
                                    # Print the progress bar on a new line
                                    progress_message += f"[{bar}] {current[i]}/{total[i]} ({percentage:.2f}%) {names[i]}        "

                                if current[0] % 5000 == 0 and current[0]>0:
                                    elapsed = time.time() - start_data_time
                                    print(f"{progress_message}")
                                    #sys.exit()
                                    file_size = get_file_size_in_mb(f"{data_folder}/data.h5")
                                    print(f"file size: {file_size:.4f}MB")
                                    print(f"{elapsed:.4f}s taken to write 5000 data points.")
                                    start_data_time = time.time()
                                    print('\n\n')
                                
                                piece_stats = get_piece_stats(chess.Board(fens[k]))
                                
                                if piece_stats['total_pieces'] >= 26:
                                    if opening_count < max_openings:
                                        row['phase'] = phase_encoder['opening']
                                        encoded_row['phase'] = phase_encoder['opening']
                                        opening_count+=1
                                    else:
                                        continue
                                elif piece_stats['total_pieces']<=26 and piece_stats['total_pieces']>=14:
                                    if middlegame_count < max_middlegames:
                                        row['phase'] = phase_encoder['middlegame']
                                        encoded_row['phase'] = phase_encoder['middlegame']
                                        middlegame_count+=1
                                    else:
                                        continue
                                else:
                                    if endgame_count < max_endgames:
                                        row['phase'] = phase_encoder['endgame']
                                        encoded_row['phase'] = phase_encoder['endgame']
                                        endgame_count+=1
                                    else:
                                        continue
                                
                                t, b, wk, wq, bk, bq = parse_fen(fens[k])
                                ms = (
                                    ["<move>"]
                                    + moves[k : k + max_move_sequence_length]
                                    + ["<pad>"] * ((k + max_move_sequence_length) - len(moves))
                                )
                                msl = len([m for m in ms if m != "<pad>"]) - 1

                                # Board position
                                row["board_position"] = b
                                encoded_row["board_position"] = encode(b, PIECES)
                                
                                #row['raw_fen'] = fens[k]

                                # Turn
                                row["turn"] = t
                                encoded_row["turn"] = encode(t, TURN)

                                # Castling rights
                                row["white_kingside_castling_rights"] = wk
                                row["white_queenside_castling_rights"] = wq
                                row["black_kingside_castling_rights"] = bk
                                row["black_queenside_castling_rights"] = bq
                                encoded_row["white_kingside_castling_rights"] = encode(
                                    wk,
                                    BOOL,
                                )
                                encoded_row["white_queenside_castling_rights"] = encode(
                                    wq,
                                    BOOL,
                                )
                                encoded_row["black_kingside_castling_rights"] = encode(
                                    bk,
                                    BOOL,
                                )
                                encoded_row["black_queenside_castling_rights"] = encode(
                                    bq,
                                    BOOL,
                                )

                                # Move sequence
                                row["moves"] = ms
                                encoded_row["moves"] = encode(
                                    ms,
                                    UCI_MOVES,
                                )

                                # Move sequence lengths
                                row["length"] = msl
                                encoded_row["length"] = msl

                                # "From" and "To" squares corresponding to next move
                                row["from_square"] = ms[1][:2]
                                encoded_row["from_square"] = encode(ms[1][:2], SQUARES)
                                row["to_square"] = ms[1][2:4]
                                encoded_row["to_square"] = encode(ms[1][2:4], SQUARES)
                                
                                
                                #new features
                                row["result"] = result
                                
                                if result=='1-0':
                                    encoded_row["result"] = 1
                                elif result=='0-1':
                                    encoded_row["result"] = -1
                                else:
                                    encoded_row["result"] = 0
                                    
                                row['base_time'] = base_time
                                row['increment_time'] = increment_time
                                encoded_row['base_time'] = base_time
                                encoded_row['increment_time'] = increment_time
                                
                                
                                if t=='w':
                                    row['white_remaining_time'] = white_clock_times[move_numbers[k]]
                                    encoded_row['white_remaining_time'] = white_clock_times[move_numbers[k]]

                                    row['black_remaining_time'] = black_clock_times[min(move_numbers[k], len(black_clock_times)-1)]
                                    encoded_row['black_remaining_time'] = black_clock_times[min(move_numbers[k], len(black_clock_times)-1)]
                                    
                                    row['time_spent_on_move'] = white_elapsed_times[move_numbers[k]]
                                    
                                    if white_elapsed_times[move_numbers[k]]==0:
                                        white_elapsed_times[move_numbers[k]] = random.uniform(0.1, 0.5)
                                    
                                    if encoded_row['white_remaining_time'] == 0:
                                        encoded_row['time_spent_on_move'] = 0.005
                                    else:
                                        encoded_row['time_spent_on_move'] = white_elapsed_times[move_numbers[k]]/encoded_row['white_remaining_time']
                                
                                if t=='b':
                                    row['white_remaining_time'] = white_clock_times[move_numbers[k]]
                                    encoded_row['white_remaining_time'] = white_clock_times[move_numbers[k]]

                                    row['black_remaining_time'] = black_clock_times[min(move_numbers[k], len(black_clock_times)-1)]
                                    encoded_row['black_remaining_time'] = black_clock_times[min(move_numbers[k], len(black_clock_times)-1)]
                                    
                                    row['time_spent_on_move'] = black_elapsed_times[move_numbers[k]]
                                    
                                    if black_elapsed_times[move_numbers[k]]==0:
                                        black_elapsed_times[move_numbers[k]] = random.uniform(0.1, 0.5)
                                        
                                    if encoded_row['black_remaining_time'] == 0:
                                        encoded_row['time_spent_on_move'] = 0.005
                                    else:
                                        encoded_row['time_spent_on_move'] = black_elapsed_times[move_numbers[k]]/encoded_row['black_remaining_time']
                                
                                #multiply by 100 to make it into a percent, easier to read
                                encoded_row['time_spent_on_move'] = encoded_row['time_spent_on_move'] * 100
                                
                                
                                row['white_rating'] = white_rating
                                encoded_row['white_rating'] = white_rating
                                
                                row['black_rating'] = black_rating
                                encoded_row['black_rating'] = black_rating
                                
                                row['move_number'] = move_numbers[k]
                                encoded_row['move_number'] = move_numbers[k]
                                
                                legal_moves_board = chess.Board(fens[k])
                                num_legal_moves = len(list(legal_moves_board.legal_moves))
                                row['num_legal_moves'] = num_legal_moves
                                encoded_row['num_legal_moves'] = num_legal_moves
                                
                                row['white_material_value'] = calculate_material(fens[k], 'white')
                                encoded_row['white_material_value'] = calculate_material(fens[k], 'white')
                                
                                row['black_material_value'] = calculate_material(fens[k], 'black')
                                encoded_row['black_material_value'] = calculate_material(fens[k], 'black')
                                
                                if t=='w':
                                    row['material_difference'] = calculate_material(fens[k], 'white') - calculate_material(fens[k], 'black')
                                    encoded_row['material_difference'] = calculate_material(fens[k], 'white') - calculate_material(fens[k], 'black')
                                if t=='b':
                                    row['material_difference'] = calculate_material(fens[k], 'black') - calculate_material(fens[k], 'white')
                                    encoded_row['material_difference'] = calculate_material(fens[k], 'black') - calculate_material(fens[k], 'white')

                                row['moves_until_end'] = move_numbers[len(move_numbers)-1] - move_numbers[k]
                                encoded_row['moves_until_end'] = move_numbers[len(move_numbers)-1] - move_numbers[k]

                                # Add row
                                row.append()
                                encoded_row.append()

                        last_processed_game_index += 1  # Update the index for the next game
                        game = chess.pgn.read_game(decompressed_file)
        except ProtocolError:
            delay = random.randint(1, 10)
            print(f"ProtocolError encountered. Retrying in {delay}s...")
            print(f"Need to skip {last_processed_game_index} games")
            time.sleep(delay)
        
def main():
    parser = argparse.ArgumentParser(description='Process data for a specific epoch.')
    parser.add_argument('--epoch', type=int, required=True, help='type the epoch number')

    args = parser.parse_args()

    epoch_num = args.epoch

    # Now you can use the epoch_num variable
    print(f"Processing data for epoch {epoch_num}")
    
    epoch_dir = f"{DATA_FOlDER}/epoch_{epoch_num}"
    train_dir = f"{epoch_dir}/train_data"
    val_dir = f"{epoch_dir}/val_data"
    
    urls = {
        'epoch_1': 'https://database.lichess.org/standard/lichess_db_standard_rated_2024-08.pgn.zst',
        'epoch_2': 'https://database.lichess.org/standard/lichess_db_standard_rated_2024-09.pgn.zst',
        'epoch_3': 'https://database.lichess.org/standard/lichess_db_standard_rated_2024-10.pgn.zst',
    }
    
    #training data
    write_to_h5(
        data_folder=train_dir,
        epoch=epoch_num-1,
        exp_rows=1e7,
        type = 'training',
        url = urls[f'epoch_{epoch_num}'],
        min_rating=1600,
        max_rating=1700,
        h5_file='data.h5',
    )
    
    #Finished writing validation data
    #validation data
    write_to_h5(
        data_folder=val_dir,
        epoch=epoch_num-1,
        exp_rows=1e3,
        type = 'validation',
        url = 'https://database.lichess.org/standard/lichess_db_standard_rated_2020-04.pgn.zst',
        min_rating=1600,
        max_rating=1700,
        h5_file='data.h5',
    )
    
    sys.exit()
    

if __name__ == "__main__":
    for epoch_num in range(num_epochs):
        epoch_dir = os.path.join(DATA_FOlDER, f'epoch_{epoch_num+1}')
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
            
            train_dir = os.path.join(f'{DATA_FOlDER}/epoch_{epoch_num+1}', f'train_data')
            val_dir = os.path.join(f'{DATA_FOlDER}/epoch_{epoch_num+1}', f'val_data')
            os.makedirs(train_dir)
            os.makedirs(val_dir)
    main()