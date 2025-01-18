import chess.pgn
from tqdm import tqdm
import argparse
from tools import TURN, PIECES, SQUARES, UCI_MOVES, BOOL
from utils import is_increment, calculate_material, encode, get_move_numbers, is_increment_greater_than_base, time_to_seconds, parse_fen, extract_clock_times_from_pgn, get_piece_stats, print_progress_bar, get_file_size_in_mb, is_berserk_game
import yaml
import sys
import time
import tables as tb
import chess
import random

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
max_move_sequence_length = cfg['model']['n_moves']

phase_encoder = {
    'opening': 0,
    'middlegame': 1,
    'endgame': 2
}

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


def process_pgn(pgn_file_path=None, 
                h5_file_path=None):
    """
    Process a PGN file, reading games one by one and displaying progress with tqdm.

    Args:
        pgn_file_path (str): Path to the PGN file to process.
        h5_file_path (str): Path to the H5 file to write to.
    """
    # Open the PGN file
    with open(pgn_file_path, 'r') as pgn_file:
        # Initialize tqdm for progress tracking
        game_count = sum(1 for _ in open(pgn_file_path, 'r') if _.startswith("[Event "))
        pgn_file.seek(0)  # Reset the file pointer to the start
        pbar = tqdm(total=game_count, desc=f"Processing games from {pgn_file_path}")
        
        lines_processed = 0
        games_processed = 0
        
        while True:
            # Read one game at a time
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                h5_file.close()
                print(f"{lines_processed} datapoints saved to disk. {games_processed} games processed.")
                break
            else:
                games_processed += 1
            
            h5_file = tb.open_file(
                h5_file_path, mode="w", title=f"data file"
            )
            
            encoded_table = h5_file.create_table(
                "/", "encoded_data", EncodedChessTable, expectedrows=1e8
            )
            
            encoded_row = encoded_table.row
            
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
            
            for k in range(start_index, len(moves)):
                lines_processed += 1
                
                piece_stats = get_piece_stats(chess.Board(fens[k]))
                                
                if piece_stats['total_pieces'] >= 26:
                    encoded_row['phase'] = phase_encoder['opening']
                elif piece_stats['total_pieces']<=26 and piece_stats['total_pieces']>=14:
                    encoded_row['phase'] = phase_encoder['middlegame']
                else:
                    encoded_row['phase'] = phase_encoder['middlegame']
                    
                t, b, wk, wq, bk, bq = parse_fen(fens[k])
                ms = (
                    ["<move>"]
                    + moves[k : k + max_move_sequence_length]
                    + ["<pad>"] * ((k + max_move_sequence_length) - len(moves))
                )
                msl = len([m for m in ms if m != "<pad>"]) - 1
                
                encoded_row["board_position"] = encode(b, PIECES)
                encoded_row["turn"] = encode(t, TURN)
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
                encoded_row["moves"] = encode(
                                    ms,
                                    UCI_MOVES,
                                )
                encoded_row["length"] = msl
                encoded_row["from_square"] = encode(ms[1][:2], SQUARES)
                encoded_row["to_square"] = encode(ms[1][2:4], SQUARES)
                if result=='1-0':
                    encoded_row["result"] = 1
                elif result=='0-1':
                    encoded_row["result"] = -1
                else:
                    encoded_row["result"] = 0
                encoded_row['base_time'] = base_time
                encoded_row['increment_time'] = increment_time
                
                if t=='w':
                    encoded_row['white_remaining_time'] = white_clock_times[move_numbers[k]]
                    encoded_row['black_remaining_time'] = black_clock_times[min(move_numbers[k], len(black_clock_times)-1)]
                    if white_elapsed_times[move_numbers[k]]==0:
                        white_elapsed_times[move_numbers[k]] = random.uniform(0.1, 0.5)
                    
                    if encoded_row['white_remaining_time'] == 0:
                        encoded_row['time_spent_on_move'] = 0.005
                    else:
                        encoded_row['time_spent_on_move'] = white_elapsed_times[move_numbers[k]]/encoded_row['white_remaining_time']
                if t=='b':
                    encoded_row['white_remaining_time'] = white_clock_times[move_numbers[k]]
                    encoded_row['black_remaining_time'] = black_clock_times[min(move_numbers[k], len(black_clock_times)-1)]
                    
                    if black_elapsed_times[move_numbers[k]]==0:
                        black_elapsed_times[move_numbers[k]] = random.uniform(0.1, 0.5)
                        
                    if encoded_row['black_remaining_time'] == 0:
                        encoded_row['time_spent_on_move'] = 0.005
                    else:
                        encoded_row['time_spent_on_move'] = black_elapsed_times[move_numbers[k]]/encoded_row['black_remaining_time']           
            
                encoded_row['white_rating'] = white_rating
                encoded_row['black_rating'] = black_rating
                encoded_row['move_number'] = move_numbers[k]
                
                legal_moves_board = chess.Board(fens[k])
                num_legal_moves = len(list(legal_moves_board.legal_moves))
                encoded_row['num_legal_moves'] = num_legal_moves
                encoded_row['white_material_value'] = calculate_material(fens[k], 'white')
                encoded_row['black_material_value'] = calculate_material(fens[k], 'black')
                
                if t=='w':
                    encoded_row['material_difference'] = calculate_material(fens[k], 'white') - calculate_material(fens[k], 'black')
                if t=='b':
                    encoded_row['material_difference'] = calculate_material(fens[k], 'black') - calculate_material(fens[k], 'white')

                encoded_row['moves_until_end'] = move_numbers[len(move_numbers)-1] - move_numbers[k]
                
                encoded_row.append()
            
            # Update tqdm
            pbar.update(1)
        
        # Close the progress bar
        pbar.close()

def main():
    """
    The main function that runs the PGN processing script.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process a PGN file.")
    parser.add_argument(
        "pgn_file_path",
        type=str,
        help="Path to the PGN file to process"
    )
    parser.add_argument(
        "h5_file_path",
        type=str,
        help="Path to the H5 file to write to"
    )
    args = parser.parse_args()

    # Call the processing function with the provided file path
    process_pgn(pgn_file_path=args.pgn_file_path,
                h5_file_path=args.h5_file_path)

# Execute the main function
if __name__ == "__main__":
    main()
