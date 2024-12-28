import tensorflow as tf

import os
import yaml
import regex
from modules import BoardEncoder, MoveDecoder
from collections import Counter
import numpy as np
from pgn_parser import dataset, val_dataset, warmup_dataset
import chess
import torch
import torch.nn.functional as F
import sys

from tools import RANKS, FILES, UCI_MOVES, TURN, PIECES, SQUARES, BOOL

class ChessTransformer(tf.keras.Model):

    def __init__(self, cfg):
        super(ChessTransformer, self).__init__()

        self.code = "ED"

        self.vocab_sizes = cfg['model']['vocab_sizes']
        self.n_moves = cfg['model']['n_moves']
        self.d_model = cfg['model']['d_model']
        self.n_heads = cfg['model']['n_heads']
        self.d_queries = cfg['model']['d_queries']
        self.d_values = cfg['model']['d_values']
        self.d_inner = cfg['model']['d_inner']
        self.n_layers = cfg['model']['n_layers']
        self.dropout = cfg['model']['dropout']

        # Encoder
        self.board_encoder = BoardEncoder(
            vocab_sizes=self.vocab_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        # Decoder
        self.move_decoder = MoveDecoder(
            vocab_size=self.vocab_sizes["moves"],
            n_moves=self.n_moves,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights in the transformer model.
        """
        # Glorot uniform initialization
        for layer in self.layers:
            for weight in layer.weights:
                if len(weight.shape) > 1:  # Check for at least two dimensions
                    initializer = tf.keras.initializers.GlorotUniform()
                    weight.assign(initializer(weight.shape))
    def call(self, inputs):
        # Encoder
        boards = self.board_encoder(
            inputs["turns"],
            inputs["white_kingside_castling_rights"],
            inputs["white_queenside_castling_rights"],
            inputs["black_kingside_castling_rights"],
            inputs["black_queenside_castling_rights"],
            inputs["board_positions"],
        )  # (N, BOARD_STATUS_LENGTH, d_model)

        # Decoder
        moves = self.move_decoder(
            inputs["encoded_moves"][:, :-1], inputs["lengths"], boards
        )  # (N, n_moves, move_vocab_size)
        # Note: We don't pass the last move as it has no next-move

        return moves
    

    
with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
model = ChessTransformer(cfg)


def replace_number(match):
    """
    Replaces numbers in a string with as many periods.

    For example, "3" will be replaced by "...".

    Args:

        match (regex.match): A RegEx match for a number.

    Returns:

        str: The replacement string.
    """
    return int(match.group()) * "."


def square_index(square):
    """
    Gets the index of a chessboard square, counted from the top-left
    corner (a8) of the chessboard.

    Args:

        square (str): The square.

    Returns:

        int: The index for this square.
    """
    file = square[0]
    rank = square[1]

    return (7 - RANKS.index(rank)) * 8 + FILES.index(file)


def assign_ep_square(board, ep_square):
    """
    Notate a board position with an En Passan square.

    Args:

        board (str): The board position.

        ep_square (str): The En Passant square.

    Returns:

        str: The modified board position.
    """
    i = square_index(ep_square)

    return board[:i] + "," + board[i + 1 :]


def get_castling_rights(castling_rights):
    """
    Get individual color/side castling rights from the FEN notation of
    castling rights.

    Args:

        castling_rights (str): The castling rights component of the FEN
        notation.

    Returns:

        bool: Can white castle kingside?

        bool: Can white castle queenside?

        bool: Can black castle kingside?

        bool: Can black castle queenside?
    """
    white_kingside = "K" in castling_rights
    white_queenside = "Q" in castling_rights
    black_kingside = "k" in castling_rights
    black_queenside = "q" in castling_rights

    return white_kingside, white_queenside, black_kingside, black_queenside


def parse_fen(fen):
    """
    Parse the FEN notation at a given board position.

    Args:

        fen (str): The FEN notation.

    Returns:

        str: The player to move next, one of "w" or "b".

        str: The board position.

        bool: Can white castle kingside?

        bool: Can white castle queenside?

        bool: Can black castle kingside?

        bool: Can black castle queenside?
    """
    board, turn, castling_rights, ep_square, _, __ = fen.split()
    board = regex.sub(r"\d", replace_number, board.replace("/", ""))
    if ep_square != "-":
        board = assign_ep_square(board, ep_square)
    (
        white_kingside,
        white_queenside,
        black_kingside,
        black_queenside,
    ) = get_castling_rights(castling_rights)

    return turn, board, white_kingside, white_queenside, black_kingside, black_queenside

def encode(item, vocabulary):
    """
    Encode an item with its index in the vocabulary its from.

    Args:

        item (list, str, bool): The item.

        vocabulary (dict): The vocabulary.

    Raises:

        NotImplementedError: If the item is not one of the types
        specified above.

    Returns:

        list, str: The item, encoded.
    """
    if isinstance(item, list):  # move sequence
        return [vocabulary[it] for it in item]
    elif isinstance(item, str):  # turn or board position or square
        return (
            vocabulary[item] if item in vocabulary else [vocabulary[it] for it in item]
        )
    elif isinstance(item, bool):  # castling rights
        return vocabulary[item]
    else:
        raise NotImplementedError
    
def get_model_inputs(board):
    """
    Get inputs to be fed to a model.

    Args:
        board (chess.Board): The chessboard in its current state.

    Returns:
        dict: The inputs to be fed to the model.
    """
    model_inputs = dict()

    t, b, wk, wq, bk, bq = parse_fen(board.fen())
    
    model_inputs["turns"] = tf.expand_dims(tf.convert_to_tensor([encode(t, vocabulary=TURN)], dtype=tf.int32), axis=0)
    model_inputs["board_positions"] = tf.expand_dims(tf.convert_to_tensor(encode(b, vocabulary=PIECES), dtype=tf.int32), axis=0)
    model_inputs["white_kingside_castling_rights"] = tf.expand_dims(tf.convert_to_tensor([encode(wk, vocabulary=BOOL)], dtype=tf.int32), axis=0)
    model_inputs["white_queenside_castling_rights"] = tf.expand_dims(tf.convert_to_tensor([encode(wq, vocabulary=BOOL)], dtype=tf.int32), axis=0)
    model_inputs["black_kingside_castling_rights"] = tf.expand_dims(tf.convert_to_tensor([encode(bk, vocabulary=BOOL)], dtype=tf.int32), axis=0)
    model_inputs["black_queenside_castling_rights"] = tf.expand_dims(tf.convert_to_tensor([encode(bq, vocabulary=BOOL)], dtype=tf.int32), axis=0)
    
    model_inputs["encoded_moves"] = tf.expand_dims(tf.convert_to_tensor([UCI_MOVES["<move>"], UCI_MOVES["<pad>"]], dtype=tf.int64), axis=0)
    model_inputs["lengths"] = tf.expand_dims(tf.convert_to_tensor([1], dtype=tf.int64), axis=0)
    return model_inputs

def topk_sampling(logits, k=1):
    """
    Randomly sample from the multinomial distribution formed from the
    "top-k" logits only.

    Args:

        logits (torch.FloatTensor): Predicted logits, of size (N,
        vocab_size).

        k (int, optional): Value of "k". Defaults to 1.

    Returns:

        torch.LongTensor: Samples (indices), of size (N).
    """
    k = min(k, logits.shape[1])

    with torch.no_grad():
        min_topk_logit_values = logits.topk(k=k, dim=1)[0][:, -1:]  # (N, 1)
        logits[logits < min_topk_logit_values] = -float("inf")  #  (N, vocab_size)
        probabilities = F.softmax(logits, dim=1)  #  (N, vocab_size)
        samples = torch.multinomial(probabilities, num_samples=1).squeeze(1)  #  (N)

    return samples

def get_move(index):
    print(index)
    # Reverse the SQUARES dictionary
    for square, idx in UCI_MOVES.items():
        if idx == index:
            return square
    return None

"""import h5py

def print_weights(group, prefix=""):
    for name in group.keys():
        item = group[name]
        if isinstance(item, h5py.Group):
            print(f"{prefix}Group: {name}")
            # Recursively explore the subgroup
            print_weights(item, prefix + "  ")
        elif isinstance(item, h5py.Dataset):
            weights = item[...]
            print(f"{prefix}{name}: shape {weights.shape}")
            print(weights)

# Open the HDF5 file
with h5py.File('model_weights/model_weights_step_4750.weights.h5', 'r') as f:
    # List all top-level groups in the file
    print("Keys in the HDF5 file:")
    for key in f.keys():
        print(key)

    # Access and print weights for each layer
    for layer_name in f.keys():
        layer = f[layer_name]  # Get the layer group
        print(f"Weights for layer '{layer_name}':")
        print_weights(layer)

"""

for i, batch in enumerate(warmup_dataset):
    model(batch[0])
    break
        
model.load_weights('model_weights/model_weights_step_22500.weights.h5')
print("model loaded")


"""-------------Validation------------"""
from loss import LabelSmoothedCE
from utils import topk_accuracy2

total = 500
top1_val = 0
top3_val = 0
top5_val = 0

criterion = LabelSmoothedCE(eps=cfg['training']['label_smoothing'],
                       n_predictions=cfg['training']['n_moves'])

for batch in val_dataset:
            batch = batch[0]
            predicted_moves = model(batch, training=False)  # (N, n_moves, move_vocab_size)

            # Loss
            with tf.device('/CPU:0'):
                loss = criterion(
                    y_true=batch,  # batch["encoded_moves"][:, 1:]  # (N, n_moves)
                    y_pred=predicted_moves,  # (N, n_moves, move_vocab_size)
                    # lengths=batch["lengths"],  # (N, 1)
                )
            lengths_sum = tf.reduce_sum(batch["lengths"])  # Sum the tensor
            lengths_sum_value = float(lengths_sum.numpy())
            # Compute accuracies on CPU to save GPU memory
            with tf.device('/CPU:0'):
                top1_accuracy_val, top3_accuracy_val, top5_accuracy_val = topk_accuracy2(
                    logits=predicted_moves[:, 0, :],
                    targets=batch["encoded_moves"][:, 1],
                    k=[1, 3, 5],
                )
            top1_val += top1_accuracy_val
            top3_val += top3_accuracy_val
            top5_val += top5_accuracy_val
            
print("top 1 validation: ",top1_val.numpy())
print("top 3 validation: ",top3_val.numpy())
print("top 5 validation: ",top5_val.numpy())
print()

"""-------------Validation------------"""



# Initialize the chess board
board = chess.Board()

while not board.is_game_over():
    print(board)
    print("Legal moves:", [move.uci() for move in board.legal_moves])

    # Player's turn
    player_move = input("Enter your move (in UCI format): ")
    
    if player_move in [move.uci() for move in board.legal_moves]:
        board.push(chess.Move.from_uci(player_move))
    else:
        print("Illegal move! Please try again.")
        continue

    # Model's turn
    input_data = get_model_inputs(board)
    predictions = model(input_data, training=False)
    torch_predictions = torch.tensor(predictions.numpy())
    predicted_moves = torch_predictions[:, 0, :]

    legal_moves = [move.uci() for move in board.legal_moves]
    legal_move_indices = [UCI_MOVES[m] for m in legal_moves]

    k = 3
    legal_move_index = topk_sampling(
        logits=predicted_moves[:, legal_move_indices],
        k=k,
    ).item()
    model_move = legal_moves[legal_move_index]

    # Make the model's move
    board.push(chess.Move.from_uci(model_move))
    print(f"Model plays: {model_move}")

# End of the game
print("Game over!")

"""max = -100
for legal_move_index in legal_move_indices:
    if predicted_moves[legal_move_index] > max:
        max = legal_move_index
print(get_move(max))"""

#legal_move_indices_tensor = tf.convert_to_tensor(legal_move_indices, dtype=tf.int32)

# Use tf.gather to index into predicted_moves
#print(predicted_moves.numpy()[0])
#logits = tf.gather(predicted_moves, legal_move_indices_tensor, axis=1)
#print(np.argmax(logits))


"""k=10
#print(predicted_moves)
legal_move_indices_tensor = tf.convert_to_tensor(legal_move_indices, dtype=tf.int32)

# Use tf.gather to index into predicted_moves
logits = tf.gather(predicted_moves, legal_move_indices_tensor, axis=1)
legal_move_index = topk_sampling(
                logits=logits,#predicted_moves[:, legal_move_indices],
                k=k,
            ).item()
model_move = legal_moves[legal_move_index]
print(model_move)"""
sys.exit()


moves = []
for move in predictions[0]:
    index = np.argmax(move)
    print(index)
    pred_move = get_move(index)
    moves.append(pred_move)
print(moves)