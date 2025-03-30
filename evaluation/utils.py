import torch
import pathlib
import chess
import regex
import argparse
from configs import import_config
import torch.nn.functional as F
import numpy as np
from configs.models.utils.levels import TURN, PIECES, UCI_MOVES, BOOL, SQUARES, FILES, RANKS
from model import ChessTemporalTransformerEncoder, PonderingTimeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_move(board, predictions):
    legal_moves = [move.uci() for move in board.legal_moves]
    predicted_from_squares = predictions['from_squares']
    predicted_to_squares = predictions['to_squares']
    predicted_from_squares = predicted_from_squares[:, 0, :]  # (1, 64)
    predicted_to_squares = predicted_to_squares[:, 0, :]  # (1, 64)
    
    # Convert "From" and "To" square predictions to move predictions
    predicted_from_log_probabilities = torch.log_softmax(
        predicted_from_squares, dim=-1
    ).unsqueeze(
        2
    )  # (1, 64, 1)
    predicted_to_log_probabilities = torch.log_softmax(
        predicted_to_squares, dim=-1
    ).unsqueeze(
        1
    )  # (1, 1, 64)
    predicted_moves = (
        predicted_from_log_probabilities + predicted_to_log_probabilities
    ).view(
        1, -1
    )  # (1, 64 * 64)

    # Filter out move indices corresponding to illegal moves
    legal_moves = list(
        set([m[:4] for m in legal_moves])
    )  # for handing pawn promotions manually, remove pawn promotion targets
    legal_move_indices = list()
    for m in legal_moves:
        from_square = m[:2]
        to_square = m[2:4]
        legal_move_indices.append(
            SQUARES[from_square] * 64 + SQUARES[to_square]
        )

    k=1
    # Perform top-k sampling to obtain a legal predicted move
    legal_move_index = topk_sampling(
        logits=predicted_moves[:, legal_move_indices],
        k=1,
    ).item()
    model_move = legal_moves[legal_move_index]

    # Handle pawn promotion manually if "model_move" is a pawn promotion move
    if is_pawn_promotion(board, model_move):
        model_move = model_move + "q"  # always promote to a queen
        
    return model_move



def is_pawn_promotion(board, move):
    """
    Check if a move (in UCI notation) corresponds to a pawn promotion on
    the given board.

    Args:

        board (chess.Board): The chessboard in its current state.

        move (str): The move un UCI notation.

    Returns:

        bool: Is the move a pawn promotion?
    """

    m = chess.Move.from_uci(move)
    if board.piece_type_at(m.from_square) == chess.PAWN and chess.square_rank(
        m.to_square
    ) in [0, 7]:
        return True
    else:
        return False


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
    board, turn, castling_rights, ep_square, halfmove_count, fullmove_count = fen.split()
    board = regex.sub(r"\d", replace_number, board.replace("/", ""))
    if ep_square != "-":
        board = assign_ep_square(board, ep_square)
    (
        white_kingside,
        white_queenside,
        black_kingside,
        black_queenside,
    ) = get_castling_rights(castling_rights)

    return turn, board, white_kingside, white_queenside, black_kingside, black_queenside, int(halfmove_count), int(fullmove_count)

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

def load_model(CONFIG):
    """
    Load model for inference.

    Args:

        CONFIG (dict): The configuration of the model.

    Returns:

        torch.nn.Module: The model.
    """
    # Model
    model = None
    if 'time' in CONFIG.NAME:
        _model = PonderingTimeModel(CONFIG, DEVICE).to(DEVICE)
    else:
        _model = ChessTemporalTransformerEncoder(CONFIG, DEVICE).to(DEVICE)

    checkpoint_path = ''
    if DEVICE.type == 'cpu':
        if 'time' in CONFIG.NAME:
            checkpoint_path = 'checkpoints/1900_step_10000.pt'
        elif 'opening' in CONFIG.NAME:
            checkpoint_path = 'checkpoints/1900_step_16000.pt'
        else:
            checkpoint_path = 'checkpoints/1900_step_30000.pt'
    else:
        checkpoint_path = '../../drive/My Drive/CT-EFT-85.pt'
    

    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), weights_only=False, map_location=torch.device('cpu'))
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '')  # remove the '_orig_mod' prefix
        new_state_dict[new_key] = value
    
    _model.load_state_dict(new_state_dict)

    # Compile model
    model = torch.compile(
        _model,
        mode=CONFIG.COMPILATION_MODE,
        dynamic=CONFIG.DYNAMIC_COMPILATION,
        disable=CONFIG.DISABLE_COMPILATION,
    )
    model = _model
    model.eval()  # eval mode disables dropout

    print(f"\n {CONFIG.NAME} Model loaded!\n")

    return model

def get_model_inputs(board,
                     time_control='600+0',
                     white_remaining_time=600,
                     black_remaining_time=600,
                     white_rating=1642,
                     black_rating=1613,
    ):
    """
    Get inputs to be fed to a model.

    Args:

        board (chess.Board): The chessboard in its current state.

    Returns:

        dict: The inputs to be fed to the model.
    """
    model_inputs = dict()
    
    phase_encoder = {
        'opening': 0,
        'middlegame': 1,
        'endgame': 2
    }

    t, b, wk, wq, bk, bq, halfmove_count, fullmove_count = parse_fen(board.fen())

    model_inputs["turn"] = (
        torch.IntTensor([encode(t, vocabulary=TURN)]).unsqueeze(0).to(DEVICE)
    )
    model_inputs["board_position"] = (
        torch.IntTensor(encode(b, vocabulary=PIECES)).unsqueeze(0).to(DEVICE)
    )
    model_inputs["white_kingside_castling_rights"] = (
        torch.IntTensor([encode(wk, vocabulary=BOOL)]).unsqueeze(0).to(DEVICE)
    )
    model_inputs["white_queenside_castling_rights"] = (
        torch.IntTensor([encode(wq, vocabulary=BOOL)]).unsqueeze(0).to(DEVICE)
    )
    model_inputs["black_kingside_castling_rights"] = (
        torch.IntTensor([encode(bk, vocabulary=BOOL)]).unsqueeze(0).to(DEVICE)
    )
    model_inputs["black_queenside_castling_rights"] = (
        torch.IntTensor([encode(bq, vocabulary=BOOL)]).unsqueeze(0).to(DEVICE)
    )
    model_inputs["moves"] = (
        torch.LongTensor(
            [
                UCI_MOVES["<move>"],
                UCI_MOVES["<pad>"],
            ]
        )
        .unsqueeze(0)
        .to(DEVICE)
    )
    model_inputs["lengths"] = torch.LongTensor([1]).unsqueeze(0).to(DEVICE)
    
    
    phase = 0
    piece_stats = get_piece_stats(board)
                                
    if piece_stats['total_pieces'] >= 26:
        phase = phase_encoder['opening']
    elif piece_stats['total_pieces']<=26 and piece_stats['total_pieces']>=14:
        phase = phase_encoder['middlegame']
    else:
        phase = phase_encoder['endgame']
    
    model_inputs["phase"] = torch.IntTensor(
            [phase]
        ).unsqueeze(0)
    model_inputs["white_remaining_time"] = torch.FloatTensor(
            [white_remaining_time]
        ).unsqueeze(0)
    model_inputs["black_remaining_time"] = torch.FloatTensor(
            [black_remaining_time]
        ).unsqueeze(0)
    model_inputs["white_rating"] = torch.FloatTensor(
            [white_rating]
        ).unsqueeze(0)
    model_inputs["black_rating"] = torch.FloatTensor(
            [black_rating]
        ).unsqueeze(0)
    model_inputs["move_number"] = torch.FloatTensor(
            [fullmove_count]
        ).unsqueeze(0)
    legal_moves_board = board
    num_legal_moves = len(list(legal_moves_board.legal_moves))
    model_inputs["num_legal_moves"] = torch.FloatTensor(
            [num_legal_moves]
        ).unsqueeze(0)
    model_inputs["white_material_value"] = torch.FloatTensor(
            [calculate_material(board.fen(), 'white')]
        ).unsqueeze(0)
    model_inputs["black_material_value"] = torch.FloatTensor(
            [calculate_material(board.fen(), 'black')]
        ).unsqueeze(0)
    
    if t=='w':
        model_inputs["material_difference"] = calculate_material(board.fen(), 'white') - calculate_material(board.fen(), 'black')
    if t=='b':
        model_inputs["material_difference"] = calculate_material(board.fen(), 'black') - calculate_material(board.fen(), 'white')
    model_inputs["material_difference"] = torch.FloatTensor(
            [model_inputs["material_difference"]]
        ).unsqueeze(0)
    
    model_inputs["base_time"] = torch.FloatTensor(
            [int(time_control.split("+")[0])]
        ).unsqueeze(0)
    model_inputs["increment_time"] = torch.FloatTensor(
            [int(time_control.split("+")[1])]
        ).unsqueeze(0)
    
    return model_inputs, fullmove_count

#SQUARES = {chess.square_name(i): i for i in range(64)}  # Map square names to indices
SQUARE_NAMES = {v: k for k, v in SQUARES.items()}

def get_all_move_probabilities(board, predictions):
    predicted_from_squares = predictions['from_squares']
    predicted_to_squares = predictions['to_squares']
    
    # Extract probabilities (assume batch size = 1, take first batch)
    predicted_from_squares = predicted_from_squares[:, 0, :].squeeze(0)  # Shape: (64,)
    predicted_to_squares = predicted_to_squares[:, 0, :].squeeze(0)  # Shape: (64,)

    # Apply softmax to get probabilities
    predicted_from_probs = torch.softmax(predicted_from_squares, dim=-1)  # (64,)
    predicted_to_probs = torch.softmax(predicted_to_squares, dim=-1)  # (64,)

    # Sort squares by probability (highest first)
    from_sorted_indices = torch.argsort(predicted_from_probs, descending=True)
    to_sorted_indices = torch.argsort(predicted_to_probs, descending=True)

    # Create a dictionary for all 64 × 64 moves
    move_probabilities = {}
    prob_sum = 0
    for i in range(64):
        from_square = SQUARE_NAMES[from_sorted_indices[i].item()]  # Get square name
        to_square = SQUARE_NAMES[to_sorted_indices[i].item()]  # Get square name
        move = from_square + to_square  # UCI move format
        prob = predicted_from_probs[from_sorted_indices[i]].item() * predicted_to_probs[to_sorted_indices[i]].item()
        move_probabilities[move] = prob
        prob_sum += prob

    # Sort the dictionary by probability in descending order
    sorted_move_probabilities = dict(sorted(move_probabilities.items(), key=lambda item: item[1], reverse=True))

    return sorted_move_probabilities




def get_move_probabilities(board, predictions):
    legal_moves = [move.uci() for move in board.legal_moves]
    predicted_from_squares = predictions['from_squares']
    predicted_to_squares = predictions['to_squares']
    predicted_from_squares = predicted_from_squares[:, 0, :]  # (1, 64)
    predicted_to_squares = predicted_to_squares[:, 0, :]  # (1, 64)
    
    # Convert "From" and "To" square predictions to move predictions
    predicted_from_log_probabilities = torch.log_softmax(
        predicted_from_squares, dim=-1
    ).unsqueeze(
        2
    )  # (1, 64, 1)
    predicted_to_log_probabilities = torch.log_softmax(
        predicted_to_squares, dim=-1
    ).unsqueeze(
        1
    )  # (1, 1, 64)
    predicted_moves = (
        predicted_from_log_probabilities + predicted_to_log_probabilities
    ).view(
        1, -1
    )  # (1, 64 * 64)

    # Filter out move indices corresponding to illegal moves
    legal_moves = list(
        set([m[:4] for m in legal_moves])
    )  # for handling pawn promotions manually, remove pawn promotion targets
    legal_move_indices = list()
    for m in legal_moves:
        from_square = m[:2]
        to_square = m[2:4]
        legal_move_indices.append(
            SQUARES[from_square] * 64 + SQUARES[to_square]
        )
    
    # Create dictionary for legal moves and their probabilities
    move_probabilities = {}
    for idx in legal_move_indices:
        move = legal_moves[legal_move_indices.index(idx)]
        prob = torch.exp(predicted_moves[0, idx]).item()  # Exponentiate the log-probability
        move_probabilities[move] = prob

    # Sort the dictionary by probability in descending order
    sorted_move_probabilities = dict(sorted(move_probabilities.items(), key=lambda item: item[1], reverse=True))

    return sorted_move_probabilities

def convert_uci_to_san(uci_moves, board):
    san_moves = {}
    for uci_move, probability in uci_moves.items():
        move = chess.Move.from_uci(uci_move)  # Convert UCI to Move object
        if move in board.legal_moves:
            san_move = board.san(move)  # Convert Move object to SAN
            san_moves[san_move] = probability  # Save the SAN move with its probability
    return san_moves




if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name)
    CONFIG = CONFIG.CONFIG()

    # Train model
    model = load_model(CONFIG)
    pondering_time_model = load_model(import_config('pondering_time_model').CONFIG())
    board = chess.Board("4B1k1/R4p1p/4p3/p4p2/1bP5/1P4PP/5K2/8 b - - 0 34")
    white_remaining_time=50
    black_remaining_time=50
    white_rating = 1950
    black_rating=1950
    time_control = '180+0'
    
    inputs = get_model_inputs(board,
                            time_control=time_control,
                            white_remaining_time=white_remaining_time,
                            black_remaining_time=black_remaining_time,
                            white_rating=white_rating,
                            black_rating=black_rating)
    predictions = model(inputs)
    pondering_time_pred = pondering_time_model(inputs) 
    predictions['move_time'] = pondering_time_pred['move_time'][0].item()
    model_move = get_move(board, predictions)
    print(get_move_probabilities(board, predictions))
    
    print(board)
    print("FEN: ", board.fen())
    print("Time control: ", time_control)
    print(f"White remaining time: {white_remaining_time}s")
    print(f"Black remaining time: {black_remaining_time}s")
    print(f"White rating: {white_rating}")
    print(f"Black rating: {black_rating}")
    
    print("\n\n")   
    print("predicted move: ",model_move)
    
    if predictions['move_time'] is not None:
        if board.turn:
            print(f"Predicted time for white to spend on move {round(predictions['move_time'], 4)}s")
        else:
            print(f"Predicted time for black to spend on move {round(predictions['move_time'], 4)}s")
    #print("Model's evaluation of the position is: ", round(predictions['game_result'][0].item(), 4))
    #print(f"Predicted number of full moves until the game ends: {int(predictions['moves_until_end'][0].item()*100)}")
    print("Probability that white wins: ", round(predictions['categorical_game_result'][0][2].item(), 4))
    print("Probability of a draw: ", round(predictions['categorical_game_result'][0][1].item(), 4))
    print("Probability that black wins: ", round(predictions['categorical_game_result'][0][0].item(), 4))