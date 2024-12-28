import os
import regex
import numpy as np
from collections import Counter
import chess
import re

from tools import RANKS, FILES

import tensorflow as tf



def calculate_material(fen, color):
    """
    Calculate the total material value of the pieces for the specified color.

    :param fen: The FEN string representing the chess board.
    :param color: The color to calculate material for, either 'white' or 'black'.
    :return: Total material value for the specified color.
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # The king is not part of material value
    }
    
    # Create a chess board from the FEN string
    board = chess.Board(fen)
    
    # Ensure the color is valid
    if color not in ['white', 'black']:
        raise ValueError("Color must be either 'white' or 'black'")
    
    # Convert color argument to chess constants
    color_enum = chess.WHITE if color == 'white' else chess.BLACK
    
    # Initialize the total material value for the specified color
    material_value = 0
    
    # Iterate over all the squares on the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color_enum:  # Check if the piece belongs to the specified color
            # Add the material value of the piece to the total
            material_value += piece_values[piece.piece_type]
    
    return material_value

# Function to check if increment time is greater than base time
def is_increment(game):
    # Extract the TimeControl header (e.g., "300+0" or "5+3")
    time_control = game.headers.get("TimeControl", "")
    
    if time_control:
        # Split the base time and increment time
        try:
            base_time, increment_time = time_control.split('+')
            base_time = int(base_time)  # Convert base time to integer
            increment_time = int(increment_time)  # Convert increment time to integer
            
            # Check if there is no increment
            return increment_time != 0 or base_time<600
        except ValueError:
            # Handle the case where TimeControl is not in the expected format
            print(f"Invalid TimeControl format: {time_control}")
            return False
    return False

# Function to check if increment time is greater than base time
def is_increment_greater_than_base(game):
    # Extract the TimeControl header (e.g., "300+0" or "5+3")
    time_control = game.headers.get("TimeControl", "")
    
    if time_control:
        # Split the base time and increment time
        try:
            base_time, increment_time = time_control.split('+')
            base_time = int(base_time)  # Convert base time to integer
            increment_time = int(increment_time)  # Convert increment time to integer
            
            # Check if increment time is greater than base time
            return increment_time > base_time
        except ValueError:
            # Handle the case where TimeControl is not in the expected format
            print(f"Invalid TimeControl format: {time_control}")
            return False
    return False

def is_berserk_game(game):
    time_control = game.headers.get("TimeControl", "")
    
    if time_control:
        # Split the base time and increment time
        try:
            base_time, increment_time = time_control.split('+')
            base_time = int(base_time)  # Convert base time to integer
            increment_time = int(increment_time)  # Convert increment time to integer
            
            white_clock_times = []
            black_clock_times = []

            # Iterate through all the moves in the game
            for node in game.mainline():
                # Check if the node contains a clock time (if the move has a comment with %clk)
                clock_time = node.clock()
                
                # If clock time is found, append to the correct list based on the player's turn
                if clock_time is not None:
                    if node.turn() == False:
                        #False means white's turn
                        white_clock_times.append(float(clock_time))
                    else:
                        black_clock_times.append(float(clock_time))
                        
            if len(white_clock_times)<2 or len(black_clock_times)<2:
                return False
            
            # Check if increment time is greater than base time
            return base_time*0.5 == white_clock_times[0] or base_time*0.5 == black_clock_times[0]
        except ValueError:
            # Handle the case where TimeControl is not in the expected format
            print(f"Invalid TimeControl format: {time_control}")
            return False
    return False

def topk_accuracy2(logits, targets, other_logits=None, other_targets=None, k=[1, 3, 5]):
    
    """Compute "top-k" accuracies for multiple values of "k".

    Optionally, a second set of logits and targets can be provided. In this case,
    probabilities associated with both sets of logits are combined to arrive at the
    best combinations of both predicted variables.

    Args:
        logits (tf.Tensor): Predicted logits, of size (N, vocab_size).
        targets (tf.Tensor): Actual targets, of size (N).
        other_logits (tf.Tensor, optional): Predicted logits for a second predicted variable,
            if any, of size (N, other_vocab_size). Defaults to None.
        other_targets (tf.Tensor, optional): Actual targets for a second predicted variable,
            if any, of size (N). Defaults to None.
        k (list, optional): Values of "k". Defaults to [1, 3, 5].

    Returns:
        list: "Top-k" accuracies."""
    
    batch_size = tf.shape(logits)[0]
    #print("logits:",logits)
    #print("targets:",targets)

    if other_logits is not None:
        # Get probabilities
        probabilities = tf.nn.softmax(logits)  # (N, vocab_size)
        other_probabilities = tf.nn.softmax(other_logits)  # (N, other_vocab_size)

        # Combine probabilities
        combined_probabilities = tf.matmul(probabilities, tf.expand_dims(other_probabilities, axis=1))  # (N, vocab_size, 1)
        combined_probabilities = tf.reshape(combined_probabilities, (batch_size, -1))  # (N, vocab_size * other_vocab_size)

        # Get top-k indices
        flattened_indices = tf.argsort(combined_probabilities, direction='DESCENDING')[:, :max(k)]  # (N, max(k))

        indices = flattened_indices // tf.shape(other_logits)[-1]  # (N, max(k))
        other_indices = flattened_indices % tf.shape(other_logits)[-1]  # (N, max(k))

        # Expand targets
        targets_expanded = tf.expand_dims(targets, axis=1)  # (N, 1)
        targets_expanded = tf.tile(targets_expanded, [1, max(k)])  # (N, max(k))

        other_targets_expanded = tf.expand_dims(other_targets, axis=1)  # (N, 1)
        other_targets_expanded = tf.tile(other_targets_expanded, [1, max(k)])  # (N, max(k))

        # Get correct predictions
        correct_predictions = tf.equal(indices, targets_expanded) & tf.equal(other_indices, other_targets_expanded)  # (N, max(k))

    else:
        # Get top-k indices
        _, indices = tf.nn.top_k(logits, k=max(k))  # (N, max(k))
        #print("indices:",indices)

        # Expand targets
        targets_expanded = tf.expand_dims(targets, axis=1)  # (N, 1)
        targets_expanded = tf.tile(targets_expanded, [1, max(k)])  # (N, max(k))

        # Get correct predictions
        correct_predictions = tf.equal(indices, targets_expanded)  # (N, max(k))
        #print("correct predictions:",correct_predictions)

    # Calculate top-k accuracies
    topk_accuracies = [tf.reduce_sum(tf.cast(correct_predictions[:, :k_value], tf.float32)).numpy() / tf.cast(batch_size, tf.float32) for k_value in k]
    #print("top k accuracies", topk_accuracies)

    return topk_accuracies
    
def top_n_indices(arr, n):
    # Use np.argpartition to get the indices of the top n elements
    indices = np.argpartition(arr, -n)[-n:]
    # Sort the indices based on the values in arr
    sorted_indices = indices[np.argsort(arr[indices])[::-1]]
    return sorted_indices



def time_to_seconds(clock_time):
    # Split the time string into hours, minutes, and seconds
    hours, minutes, seconds = map(int, clock_time.split(':'))

    # Convert to total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds

# Function to extract clock times from a PGN file
def extract_clock_times_from_pgn(game):
    white_clock_times = []
    black_clock_times = []

    # Iterate through all the moves in the game
    for node in game.mainline():
        # Check if the node contains a clock time (if the move has a comment with %clk)
        clock_time = node.clock()
        
        # If clock time is found, append to the correct list based on the player's turn
        if clock_time is not None:
            if node.turn() == False:
                #False means white's turn
                white_clock_times.append(clock_time)
            else:
                black_clock_times.append(clock_time)

    # Initialize separate lists for White and Black elapsed times
    white_elapsed_times = []
    black_elapsed_times = []

    # Variables to store previous clock times
    previous_white_time = None
    previous_black_time = None

    # Iterate through all the moves in the game
    for node in game.mainline():
        # Check if the node contains a clock time (if the move has a comment with %clk)
        clock_time = node.clock()
        
        # If a clock time is found, calculate elapsed time
        if clock_time is not None:
            #False means white's turn
            if node.turn() == False:
                # If this is White's move, calculate the elapsed time for White
                if previous_white_time is not None:
                    elapsed_time = previous_white_time - clock_time
                    white_elapsed_times.append(elapsed_time)
                previous_white_time = clock_time  # Update the previous clock time for White
            else:
                # If this is Black's move, calculate the elapsed time for Black
                if previous_black_time is not None:
                    elapsed_time = previous_black_time - clock_time
                    black_elapsed_times.append(elapsed_time)
                previous_black_time = clock_time  # Update the previous clock time for Black

    white_elapsed_times.insert(0, 0)
    black_elapsed_times.insert(0, 0)        

    return white_clock_times, black_clock_times, white_elapsed_times, black_elapsed_times

def get_move_numbers(game):
    move_counts = []

    # Track the current move number for White and Black
    white_move_count = 0
    black_move_count = 0

    # Iterate through the moves in the game
    for i, move in enumerate(game.mainline_moves()):
        if i % 2 == 0:  # White's move (even-indexed)
            white_move_count += 1
            move_counts.append(white_move_count - 1)  # Append the move number for White
        else:  # Black's move (odd-indexed)
            black_move_count += 1
            move_counts.append(black_move_count - 1)  # Append the move number for Black

    return move_counts

# Piece values dictionary (common chess piece values)
piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # Kings are not typically valued in material count
}

def get_piece_stats(board):
    # Initialize statistics
    total_pieces = 0
    white_pieces = 0
    black_pieces = 0
    white_value = 0
    black_value = 0

    # Iterate over the piece map to count and value the pieces
    for piece in board.piece_map().values():
        total_pieces += 1
        if piece.color == chess.WHITE:
            white_pieces += 1
            white_value += piece_values[piece.piece_type]
        else:  # black pieces
            black_pieces += 1
            black_value += piece_values[piece.piece_type]
    
    return {
        "total_pieces": total_pieces,
        "white_pieces": white_pieces,
        "black_pieces": black_pieces,
        "white_value": white_value,
        "black_value": black_value
    }

import sys
def print_progress_bar(iteration, total, name=None, bar_length=50, bar_id=0):
    # Move to the correct line for this progress bar
    sys.stdout.write(f"\033[{bar_id+1}A")  # Move cursor up to the line bar_id
    # Calculate the progress as a fraction
    progress = (iteration / total)
    # Calculate the number of blocks to represent the progress
    block = int(round(bar_length * progress))
    # Create the progress bar string
    progress_bar = f"[{'#' * block}{'.' * (bar_length - block)}] {iteration}/{total} ({progress * 100:.1f}%) {name}"
    
    # Print the progress bar on the same line (using \r to return the cursor to the start)
    sys.stdout.write('\r' + progress_bar)
    sys.stdout.flush()
    
def get_file_size_in_mb(file_path):
    """
    Get the size of a file in MB.
    
    Parameters:
    - file_path: The path to the file whose size you want to get.
    
    Returns:
    - The size of the file in MB.
    """
    # Get the file size in bytes
    file_size_bytes = os.path.getsize(file_path)
    
    # Convert bytes to MB
    file_size_mb = file_size_bytes / (1024 ** 2)
    
    return file_size_mb

    
    
def topk_accuracy(pred, targets, k=[1, 3, 5], moves_to_compare=5):
    """
    Compute "top-k" accuracies for multiple values of "k".

    Optionally, a second set of logits and targets can be provided. In this case,
    probabilities associated with both sets of logits are combined to arrive at the
    best combinations of both predicted variables.

    Args:
        logits (tf.Tensor): Predicted logits, of size (N, vocab_size).
        targets (tf.Tensor): Actual targets, of size (N).
        other_logits (tf.Tensor, optional): Predicted logits for a second predicted variable,
            if any, of size (N, other_vocab_size). Defaults to None.
        other_targets (tf.Tensor, optional): Actual targets for a second predicted variable,
            if any, of size (N). Defaults to None.
        k (list, optional): Values of "k". Defaults to [1, 3, 5].

    Returns:
        list: "Top-k" accuracies.
    """
    batch_size = tf.shape(pred)[0]
    topk_accuracies = {}
    topk_accuracies_per_move = {}
    
    for i in range(1, moves_to_compare):
        topk_accuracies_per_move[f'move_{i}'] = {}
        for k_value in k:
            topk_accuracies_per_move[f'move_{i}'][f'k_{k_value}'] = 0#AverageMeter()
            
    print("initialized topk_accuracies_per_move:", topk_accuracies_per_move)
        
    
    correct = 0
    total_moves = batch_size
    
    
    for m, target in enumerate(targets):
        for i in range(1, moves_to_compare):#len(target)):
            print("target sequence:",target)
            s=[]
            for l in pred[m].numpy():
                s.append(np.argmax(l))
            s = tf.convert_to_tensor(s)
            print("predicted sequence:",s)
            for k_value in k:
                correct_move = target[i].numpy()
                if correct_move >= 1968:
                    topk_accuracies[f'k_{k_value}'] = 0.00001
                    break
                pred_moves = top_n_indices(pred[m][i-1].numpy(), k_value)
                
                for pred_move in pred_moves:
                    #print("pred", pred_move)
                    #print("correct", correct_move)
                    if pred_move==correct_move:
                        correct+=1
                print("total correct:",correct)
                #topk_accuracies[f'k_{k_value}'] = correct#(correct/total_moves).numpy()
                topk_accuracies_per_move[f'move_{i}'][f'k_{k_value}'] += correct
                #topk_accuracies_per_move[f'move_{i}'][f'k_{k_value}'].update(correct, batch_size)
                #print("top k accuracies:",topk_accuracies)
                correct = 0
            #topk_accuracies_per_move[f'move_{i}'] = topk_accuracies
            #print("top k accuracies per move:",topk_accuracies_per_move)
            #topk_accuracies = {}
            
    print("topk accuracies per move:",topk_accuracies_per_move)
    for i in range(1, moves_to_compare):
        for k_value in k:
            topk_accuracies_per_move[f'move_{i}'][f'k_{k_value}'] /= total_moves

    return topk_accuracies_per_move
    


def change_lr(optimizer, new_lr):
    """
    Change learning rate to a specified value.
    
    Args:
        optimizer (tf.keras.optimizers.Optimizer): Optimizer whose learning rate
        must be changed.
        
        new_lr (float): New learning rate.
    """
    # Create a new optimizer with the updated learning rate
    optimizer.learning_rate.assign(new_lr)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



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