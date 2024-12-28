import requests
import zstandard as zstd
import chess.pgn
import io
import tensorflow as tf
import numpy as np
import time
import sys
import os
import yaml
from utils import encode, parse_fen
from urllib3.exceptions import ProtocolError
from tools import TURN, PIECES, SQUARES, UCI_MOVES, BOOL

max_move_sequence_length = 10
min_rating = 2000
max_rating = 10000

feature_specs = (
    #Inputs 
    {
        "turns": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        "white_kingside_castling_rights": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        "white_queenside_castling_rights": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        "black_kingside_castling_rights": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        "black_queenside_castling_rights": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        "board_positions": tf.TensorSpec(shape=(64,), dtype=tf.int32),
        "lengths": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        "encoded_moves": tf.TensorSpec(shape=(max_move_sequence_length+1,), dtype=tf.int32),
    }, 
    
    #Outputs
    #"encoded_moves": tf.TensorSpec(shape=(None,), dtype=tf.int32),  # Assuming the move sequence length can vary
    {
        "lengths": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        "encoded_moves": tf.TensorSpec(shape=(max_move_sequence_length+1,), dtype=tf.int32),
        #"from_square": tf.TensorSpec(shape=(1,), dtype=tf.int32),
    }
    #"to_square": tf.TensorSpec(shape=(1,), dtype=tf.int32),
)

def get_tensor(data):
    with tf.device('/CPU:0'):
        # Extract information from data[i] and convert to TensorFlow tensors
        turns = tf.convert_to_tensor([data["turn"]], dtype=tf.int32)
        
        white_kingside_castling_rights = tf.convert_to_tensor(
            [data["white_kingside_castling_rights"]], dtype=tf.int32
        )  # (1)
        white_queenside_castling_rights = tf.convert_to_tensor(
            [data["white_queenside_castling_rights"]], dtype=tf.int32
        )  # (1)
        black_kingside_castling_rights = tf.convert_to_tensor(
            [data["black_kingside_castling_rights"]], dtype=tf.int32
        )  # (1)
        black_queenside_castling_rights = tf.convert_to_tensor(
            [data["black_queenside_castling_rights"]], dtype=tf.int32
        )  # (1)  
        
        # For board position, convert to tensor with dtype=int32
        board_position = tf.convert_to_tensor(
            data["board_position"], dtype=tf.int32
        )  # (64)
        
        # Moves array, sliced similarly as in PyTorch
        encoded_moves = tf.convert_to_tensor(
            data["encoded_moves"][:data["n_moves"] + 1], dtype=tf.int64
        )  # (n_moves + 1)
        
        # Length clamping, equivalent to PyTorch's clamp. Using tf.minimum.
        length = tf.convert_to_tensor([data["length"]], dtype=tf.int64)
        #length = tf.minimum(length, data["n_moves"])
        length = tf.clip_by_value(length, 0, data["n_moves"]) 

        """from_square = tf.convert_to_tensor(
            [data["from_square"]], dtype=tf.int64
        )  """
        
        """to_square = tf.convert_to_tensor(
            [data["to_square"]], dtype=tf.int64
        ) """
        
        # Return the dictionary with all the necessary tensors
        return (
            {
                "turns": turns,
                "white_kingside_castling_rights": white_kingside_castling_rights,
                "white_queenside_castling_rights": white_queenside_castling_rights,
                "black_kingside_castling_rights": black_kingside_castling_rights,
                "black_queenside_castling_rights": black_queenside_castling_rights,
                "board_positions": board_position,
                "lengths": length,
                "encoded_moves": encoded_moves,
                
            },
            #"encoded_moves": encoded_moves,
            {
                "lengths": length,
                "encoded_moves": encoded_moves,
                #"from_square": from_square,
            }
            #"to_square": to_square,
            
        )
        
        

def data_generator(url, is_validation=False):
    CHUNK_SIZE = None
    
    if is_validation==False:
        with open('config.yaml', 'r') as file:
            cfg = yaml.safe_load(file)
        last_processed_game_index = (cfg['training']['step']*4*cfg['dataloading']['batch_size'])//25  # Keep track of the last processed game index
    else: 
        last_processed_game_index = 0
        
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
                    print(f"time taken to skip {last_processed_game_index} games: {elapsed:.4f}s")

                    game = chess.pgn.read_game(decompressed_file)
                    while game:
                        white_rating = game.headers.get("WhiteElo")
                        black_rating = game.headers.get("BlackElo")
                        
                        if white_rating!='?' and black_rating!='?':
                            white_rating = int(white_rating)
                            black_rating = int(black_rating)
                        else:
                            continue
                        
                        # Extracting moves
                        moves = []
                        node = game
                        while node.variations:
                            move = node.variations[0].move
                            moves.append(move)
                            node = node.variations[0]

                        num_moves = len(moves)
                        
                        game_start_index = 12
                        
                        if(
                            (white_rating >= min_rating and white_rating<=max_rating)
                            and (black_rating >= min_rating and black_rating<=max_rating)
                            and (num_moves >= max_move_sequence_length)
                            #and (num_moves >= game_start_index)
                        ):
                            if CHUNK_SIZE == None:
                                
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
                                
                                result = game.headers.get("Result")
                                start_index = 0 if result == "1-0" else 1
                                #start_index += game_start_index - 1
                                
                                for k in range(start_index, len(moves), 2):
                                    t, b, wk, wq, bk, bq = parse_fen(fens[k])
                                    ms = (
                                                ["<move>"]
                                                + moves[k : k + max_move_sequence_length]
                                                + ["<pad>"] * ((k + max_move_sequence_length) - len(moves))
                                        )
                                    msl = len([m for m in ms if m != "<pad>"]) - 1
                                    board = encode(b, PIECES)
                                    turn = encode(t, TURN)
                                    white_kingside_castling_rights = encode(wk, BOOL)
                                    white_queenside_castling_rights = encode(
                                                wq,
                                                BOOL,
                                            )
                                    black_kingside_castling_rights = encode(
                                                bk,
                                                BOOL,
                                            )
                                    black_queenside_castling_rights = encode(
                                                bq,
                                                BOOL,
                                            )
                                    encoded_moves = encode(
                                                ms,
                                                UCI_MOVES,
                                            )
                                    length = msl
                                    data = get_tensor({
                                        "board_position": board, 
                                        "turn": turn, 
                                        "white_kingside_castling_rights": white_kingside_castling_rights, 
                                        "white_queenside_castling_rights": white_queenside_castling_rights,
                                        "black_kingside_castling_rights": black_kingside_castling_rights,
                                        "black_queenside_castling_rights": black_queenside_castling_rights,
                                        "encoded_moves": encoded_moves,
                                        "length": length,
                                        "n_moves":10
                                    })
                                    yield data

                        last_processed_game_index += 1  # Update the index for the next game
                        game = chess.pgn.read_game(decompressed_file)
        
        except ProtocolError:
            print("ProtocolError encountered. Retrying...")
            time.sleep(1)
                
                
# Create a tf.data.Dataset from the generator
def create_tf_dataset(url, is_validation=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(url, is_validation),
        output_signature=feature_specs
    )
    
    return dataset

with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

batch_size = cfg['dataloading']['batch_size']
dataset = create_tf_dataset('https://database.lichess.org/standard/lichess_db_standard_rated_2024-08.pgn.zst', is_validation=False)
dataset = dataset.shuffle(buffer_size=int(float(cfg['dataloading']['buffer_size']))).prefetch(tf.data.AUTOTUNE).batch(batch_size).cache()

val_dataset = create_tf_dataset('https://database.lichess.org/standard/lichess_db_standard_rated_2020-04.pgn.zst', is_validation=True)
val_dataset = val_dataset.batch(500).take(1).cache()

warmup_dataset = create_tf_dataset('https://database.lichess.org/standard/lichess_db_standard_rated_2020-04.pgn.zst', is_validation=True)
warmup_dataset = warmup_dataset.batch(1).take(1).cache()