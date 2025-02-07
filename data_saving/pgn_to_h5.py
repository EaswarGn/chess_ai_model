import chess.pgn
from tqdm import tqdm
import argparse
from tools import TURN, PIECES, SQUARES, UCI_MOVES, BOOL
from utils import calculate_material, encode, get_move_numbers, parse_fen, extract_clock_times_from_pgn, get_piece_stats, is_berserk_game
import yaml
import sys
import time
import tables as tb
import chess
import random

"""with open('../../../config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
max_move_sequence_length = cfg['model']['n_moves']"""
max_move_sequence_length=10

phase_encoder = {
    'opening': 0,
    'middlegame': 1,
    'endgame': 2
}

# Create table description for HDF5 file
class EncodedChessTable(tb.IsDescription):
    board_position = tb.Int8Col(shape=(64))#done
    turn = tb.Int8Col() #done
    white_kingside_castling_rights = tb.Int8Col()#done
    white_queenside_castling_rights = tb.Int8Col()#done
    black_kingside_castling_rights = tb.Int8Col()#done
    black_queenside_castling_rights = tb.Int8Col()#done
    moves = tb.Int16Col(shape=(max_move_sequence_length + 1))
    length = tb.Int8Col()
    from_square = tb.Int8Col()#done
    to_square = tb.Int8Col()#done
    phase = tb.Int8Col()#done
    result = tb.Int8Col()#done
    base_time = tb.Int16Col()
    increment_time = tb.Int16Col()
    white_remaining_time = tb.Float32Col()
    black_remaining_time = tb.Float32Col()
    white_rating = tb.Float32Col()#done
    black_rating = tb.Float32Col()#done
    time_spent_on_move = tb.Float32Col()
    move_number = tb.Int16Col()#done
    num_legal_moves = tb.Int16Col()#done
    white_material_value = tb.Int16Col()#done
    black_material_value = tb.Int16Col()#done
    material_difference = tb.Int16Col()#done
    moves_until_end = tb.Int16Col()#done


def process_pgn(pgn_file_path=None, 
                h5_file_path=None):
    """
    Process a PGN file, reading games one by one and displaying progress with tqdm.

    Args:
        pgn_file_path (str): Path to the PGN file to process.
        h5_file_path (str): Path to the H5 file to write to.
    """
    with open(pgn_file_path, 'r') as pgn_file:
        game_count = sum(1 for _ in open(pgn_file_path, 'r') if _.startswith("[Event "))
        pgn_file.seek(0) 
        pbar = tqdm(total=game_count, position=0, desc=f"Processing games from {pgn_file_path}")
        
        
        h5_file = tb.open_file(
                h5_file_path, mode="w", title=f"data file"
            )
            
        encoded_table = h5_file.create_table(
            "/", "encoded_data", EncodedChessTable, expectedrows=1e8
        )
        
        encoded_row = encoded_table.row
        
        lines_processed = 0
        games_processed = 0
        games_removed = 0
        start_time = time.time()
        
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                elapsed_time = time.time() - start_time
                h5_file.close()
                print()
                print(f"{lines_processed} datapoints saved to disk. {games_processed} games processed.")
                print(f"Processed in {elapsed_time:.4f}s")
                print(f"Removed {games_removed} games")
                print()
                break
            else:
                is_games_removed = False
                #loop to find a valid game
                while (game.headers.get("Event") == 'Rated Correspondence game' or
                        game.headers.get("WhiteElo") == '?' or 
                        game.headers.get("BlackElo") == '?' or
                        game.headers.get("TimeControl") == '-' or
                        game.headers.get("Termination") == 'Abandoned' or
                        is_berserk_game(game)
                    ):
                    game = chess.pgn.read_game(pgn_file)
                    games_removed+=1
                    is_games_removed = True
                    game_count -= 1
                if is_games_removed==True:
                    pbar.total = game_count
                    pbar.refresh()
                    
                games_processed += 1
                is_games_removed = False
            
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
            
            board = game.board()
            fens = [board.fen()]  # Initialize with the starting position
            moves = []

            # Iterate through the mainline moves
            #print(game)
            node = game
            start = time.time()
            while node.variations:
                comment = node.variations[0].comment
                comment = comment.replace("\n", "")
                
                move = node.variations[0].move

                if "eval" in comment:
                    fen = comment.split("] ")[2]
                else:
                    fen = comment.split("] ")[1]
                    
                if len(fen)>100:
                    fen = fen.split(" ")
                    fen = fen[0] + ' ' + fen[1] + ' ' + fen[2] + ' ' + fen[3] + ' ' + fen[4] + ' ' + fen[5] + ' '
                
                moves.append(move.uci())
                fens.append(fen)
                node = node.variations[0]
            white_clock_times, black_clock_times, white_elapsed_times, black_elapsed_times = extract_clock_times_from_pgn(game)
            move_numbers = get_move_numbers(game)
            print(time.time()-start)
            
            start_index = 0
            
            total_moves = len(moves)
            last_move_number = move_numbers[-1]
            board = chess.Board()
            for k in range(start_index, total_moves):
                lines_processed += 1
                board.push(chess.Move.from_uci(moves[k]))
                
                piece_stats = get_piece_stats(board)
                                
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
                    + ["<pad>"] * ((k + max_move_sequence_length) - total_moves)
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

                try:
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
                except IndexError:
                    print("game:",game)
                    print("move numbers:",move_numbers)
                    print("k:",k)
                    print("white clock times:",white_clock_times)
                    print("white elapsed times:",white_elapsed_times)
                    print("black clock times:",black_clock_times)
                    print("black elapsed times:",black_elapsed_times)        

                encoded_row['white_rating'] = white_rating
                encoded_row['black_rating'] = black_rating
                encoded_row['move_number'] = move_numbers[k]

                num_legal_moves = len(list(board.legal_moves))
                encoded_row['num_legal_moves'] = num_legal_moves
                material_white, material_black = calculate_material(fens[k])
                encoded_row['white_material_value'] = material_white
                encoded_row['black_material_value'] = material_black
                
                if t=='w':
                    encoded_row['material_difference'] = material_white - material_black
                if t=='b':
                    encoded_row['material_difference'] = material_black - material_white

                encoded_row['moves_until_end'] = last_move_number - move_numbers[k]
                
                encoded_row.append()
            
            pbar.update(1)
        
        pbar.close()

def main():
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

    process_pgn(pgn_file_path=args.pgn_file_path,
                h5_file_path=args.h5_file_path)

if __name__ == "__main__":
    main()
