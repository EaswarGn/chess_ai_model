import math
import torch
import argparse
from torch import nn

from configs import import_config
from modules import BoardEncoder, MoveDecoder, OGBoardEncoder

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class ChessTransformer(nn.Module):
    """
    The Chess Transformer, for predicting the next move and sequence of
    moves likely to follow.
    """

    def __init__(
        self,
        CONFIG,
    ):
        """
        Init.

        Args:

            CONFIG (dict): The configuration, containing the following
            parameters for the model:

                VOCAB_SIZES (dict): The sizes of the vocabularies of the
                Encoder sequence components and the Decoder (move)
                sequence.

                N_MOVES (int): The expected maximum length of output
                (move) sequences.

                D_MODEL (int): The size of vectors throughout the
                transformer model, i.e. input and output sizes for the
                Encoder and Decoder.

                N_HEADS (int): The number of heads in the multi-head
                attention.

                D_QUERIES (int): The size of query vectors (and also the
                size of the key vectors) in the multi-head attention.

                D_VALUES (int): The size of value vectors in the
                multi-head attention.

                D_INNER (int): An intermediate size in the position-wise
                FC.

                N_LAYERS (int): The number of [multi-head attention +
                multi-head attention + position-wise FC] layers in the
                Encoder and Decoder.

                DROPOUT (int): The dropout probability.
        """
        super(ChessTransformer, self).__init__()

        self.code = "ED"

        self.vocab_sizes = CONFIG.VOCAB_SIZES
        self.n_moves = CONFIG.N_MOVES
        self.d_model = CONFIG.D_MODEL
        self.n_heads = CONFIG.N_HEADS
        self.d_queries = CONFIG.D_QUERIES
        self.d_values = CONFIG.D_VALUES
        self.d_inner = CONFIG.D_INNER
        self.n_layers = CONFIG.N_LAYERS
        self.dropout = CONFIG.DROPOUT

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
        # Glorot uniform initialization with a gain of 1.
        for p in self.parameters():
            # Glorot initialization needs at least two dimensions on the
            # tensor
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)

        # For the embeddings, normal initialization with 0 mean and
        # 1/sqrt(d_model) S.D.
        nn.init.normal_(
            self.board_encoder.board_position_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.turn_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.white_kingside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.white_queenside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.black_kingside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.black_queenside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.positional_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.move_decoder.embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.move_decoder.positional_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )

        # Share weights between the embedding layer in the Decoder and
        # the logit layer
        self.move_decoder.fc.weight = self.move_decoder.embeddings.weight

    def forward(self, batch):
        """
        Forward prop.

        Args:

            batch (dict): A single batch, containing the following keys:

                turns (torch.LongTensor): The current turn (w/b), of
                size (N, 1).

                white_kingside_castling_rights (torch.LongTensor):
                Whether white can castle kingside, of size (N, 1).

                white_queenside_castling_rights (torch.LongTensor):
                Whether white can castle queenside, of size (N, 1).

                black_kingside_castling_rights (torch.LongTensor):
                Whether black can castle kingside, of size (N, 1).

                black_queenside_castling_rights (torch.LongTensor):
                Whether black can castle queenside, of size (N, 1).

                board_positions (torch.LongTensor): The current board
                positions, of size (N, 64).

                moves (torch.LongTensor): The move sequences, of size
                (N, n_moves).

                lengths (torch.LongTensor): The true lengths of the move
                sequences, not including <move> and <pad> tokens, of
                size (N, 1).

        Returns:

            torch.FloatTensor: The decoded next-move probabilities, of
            size (N, n_moves, vocab_size).
        """
        """# Encoder
        boards = self.board_encoder(
            batch["turns"],
            batch["white_kingside_castling_rights"],
            batch["white_queenside_castling_rights"],
            batch["black_kingside_castling_rights"],
            batch["black_queenside_castling_rights"],
            batch["board_positions"],
        )  # (N, BOARD_STATUS_LENGTH, d_model)"""
        
        boards = self.board_encoder(
            batch["turns"],
            batch["white_kingside_castling_rights"],
            batch["white_queenside_castling_rights"],
            batch["black_kingside_castling_rights"],
            batch["black_queenside_castling_rights"],
            batch["board_positions"],
            # New temporal feature parameters
            batch["time_control"],
            batch["move_number"],
            batch["num_legal_moves"],
            batch["white_remaining_time"],
            batch["black_remaining_time"],
            batch["phase"],
            batch["white_rating"],
            batch["black_rating"],
        )

        # Decoder
        moves = self.move_decoder(
            batch["moves"][:, :-1], batch["lengths"].squeeze(1), boards
        )  # (N, n_moves, move_vocab_size)
        # Note: We don't pass the last move as it has no next-move

        return moves


class ChessTransformerEncoder(nn.Module):
    """
    The Chess Transformer (Encoder only), for predicting the next move.
    """

    def __init__(
        self,
        CONFIG,
    ):
        """
        Init.

        Args:

            CONFIG (dict): The configuration, containing the following
            parameters for the model:

                VOCAB_SIZES (dict): The sizes of the vocabularies of the
                Encoder sequence components.

                D_MODEL (int): The size of vectors throughout the
                transformer model, i.e. input and output sizes for the
                Encoder.

                N_HEADS (int): The number of heads in the multi-head
                attention.

                D_QUERIES (int): The size of query vectors (and also the
                size of the key vectors) in the multi-head attention.

                D_VALUES (int): The size of value vectors in the
                multi-head attention.

                D_INNER (int): An intermediate size in the position-wise
                FC.

                N_LAYERS (int): The number of [multi-head attention +
                multi-head attention + position-wise FC] layers in the
                Encoder.

                DROPOUT (int): The dropout probability.
        """
        super(ChessTransformerEncoder, self).__init__()

        self.code = "E"

        self.vocab_sizes = CONFIG.VOCAB_SIZES
        self.d_model = CONFIG.D_MODEL
        self.n_heads = CONFIG.N_HEADS
        self.d_queries = CONFIG.D_QUERIES
        self.d_values = CONFIG.D_VALUES
        self.d_inner = CONFIG.D_INNER
        self.n_layers = CONFIG.N_LAYERS
        self.dropout = CONFIG.DROPOUT

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

        # Output linear layer that will compute logits for the
        # vocabulary
        self.fc = nn.Linear(self.d_model, self.vocab_sizes["moves"])

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights in the transformer model.
        """
        # Glorot uniform initialization with a gain of 1.
        for p in self.parameters():
            # Glorot initialization needs at least two dimensions on the
            # tensor
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)

        # For the embeddings, normal initialization with 0 mean and
        # 1/sqrt(d_model) S.D.
        nn.init.normal_(
            self.board_encoder.board_position_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.turn_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.white_kingside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.white_queenside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.black_kingside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.black_queenside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.positional_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )

    def forward(self, batch):
        """
        Forward prop.

        Args:

            batch (dict): A single batch, containing the following keys:

                turns (torch.LongTensor): The current turn (w/b), of
                size (N, 1).

                white_kingside_castling_rights (torch.LongTensor):
                Whether white can castle kingside, of size (N, 1).

                white_queenside_castling_rights (torch.LongTensor):
                Whether white can castle queenside, of size (N, 1).

                black_kingside_castling_rights (torch.LongTensor):
                Whether black can castle kingside, of size (N, 1).

                black_queenside_castling_rights (torch.LongTensor):
                Whether black can castle queenside, of size (N, 1).

                board_positions (torch.LongTensor): The current board
                positions, of size (N, 64).

        Returns:

            torch.FloatTensor: The next-move logits, of size (N, 1,
            vocab_size).
        """
        # Encoder
        boards = self.board_encoder(
            batch["turns"],
            batch["white_kingside_castling_rights"],
            batch["white_queenside_castling_rights"],
            batch["black_kingside_castling_rights"],
            batch["black_queenside_castling_rights"],
            batch["board_positions"],
        )  # (N, BOARD_STATUS_LENGTH, d_model)

        # Find logits over vocabulary at the "turn" token
        moves = self.fc(boards[:, :1, :])  # (N, 1, vocab_size)

        return moves


class ChessTransformerEncoderFT(nn.Module):
    """
    The Chess Transformer (Encoder only), for predicting the "From"
    square and the "To" square separately for the next move, instead of
    the move in UCI notation.
    """

    def __init__(
        self,
        CONFIG,
    ):
        """
        Init.

        Args:

            CONFIG (dict): The configuration, containing the following
            parameters for the model:

                VOCAB_SIZES (dict): The sizes of the vocabularies of the
                Encoder sequence components.

                D_MODEL (int): The size of vectors throughout the
                transformer model, i.e. input and output sizes for the
                Encoder.

                N_HEADS (int): The number of heads in the multi-head
                attention.

                D_QUERIES (int): The size of query vectors (and also the
                size of the key vectors) in the multi-head attention.

                D_VALUES (int): The size of value vectors in the
                multi-head attention.

                D_INNER (int): An intermediate size in the position-wise
                FC.

                N_LAYERS (int): The number of [multi-head attention +
                multi-head attention + position-wise FC] layers in the
                Encoder.

                DROPOUT (int): The dropout probability.
        """
        super(ChessTransformerEncoderFT, self).__init__()

        self.code = "EFT"

        self.vocab_sizes = CONFIG.VOCAB_SIZES
        self.d_model = CONFIG.D_MODEL
        self.n_heads = CONFIG.N_HEADS
        self.d_queries = CONFIG.D_QUERIES
        self.d_values = CONFIG.D_VALUES
        self.d_inner = CONFIG.D_INNER
        self.n_layers = CONFIG.N_LAYERS
        self.dropout = CONFIG.DROPOUT

        # Encoder
        self.board_encoder = OGBoardEncoder(
            vocab_sizes=self.vocab_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        # Output linear layers - for the "From" square and "To" square
        self.from_squares = nn.Linear(self.d_model, 1)
        self.to_squares = nn.Linear(self.d_model, 1)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights in the transformer model.
        """
        # Glorot uniform initialization with a gain of 1.
        for p in self.parameters():
            # Glorot initialization needs at least two dimensions on the
            # tensor
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)

        # For the embeddings, normal initialization with 0 mean and
        # 1/sqrt(d_model) S.D.
        nn.init.normal_(
            self.board_encoder.board_position_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.turn_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.white_kingside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.white_queenside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.black_kingside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.black_queenside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.positional_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )

    def forward(self, batch):
        """
        Forward prop.

        Args:

            batch (dict): A single batch, containing the following keys:

                turns (torch.LongTensor): The current turn (w/b), of
                size (N, 1).

                white_kingside_castling_rights (torch.LongTensor):
                Whether white can castle kingside, of size (N, 1).

                white_queenside_castling_rights (torch.LongTensor):
                Whether white can castle queenside, of size (N, 1).

                black_kingside_castling_rights (torch.LongTensor):
                Whether black can castle kingside, of size (N, 1).

                black_queenside_castling_rights (torch.LongTensor):
                Whether black can castle queenside, of size (N, 1).

                board_positions (torch.LongTensor): The current board
                positions, of size (N, 64).

        Returns:

            torch.FloatTensor: The "From" square logits, of size (N,
            64).

            torch.FloatTensor: The "To" square logits, of size (N, 64).
        """
        # Encoder
        boards = self.board_encoder(
            batch["turns"],
            batch["white_kingside_castling_rights"],
            batch["white_queenside_castling_rights"],
            batch["black_kingside_castling_rights"],
            batch["black_queenside_castling_rights"],
            batch["board_positions"],
        )  # (N, BOARD_STATUS_LENGTH, d_model)

        # Find logits over vocabulary at the "turn" token
        from_squares = (
            self.from_squares(boards[:, 5:, :]).squeeze(2).unsqueeze(1)
        )  # (N, 1, 64)
        to_squares = (
            self.to_squares(boards[:, 5:, :]).squeeze(2).unsqueeze(1)
        )  # (N, 1, 64)

        return from_squares, to_squares
    
class ResidualBlock(nn.Module):
    def __init__(self, in_features, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return x + self.block(x)
    
class ChessTemporalTransformerEncoder(nn.Module):
    """
    Extended Chess Transformer Encoder with additional prediction heads:
    1. From and To square prediction
    2. Game result prediction (white win/black win)
    3. Move time prediction
    """

    def __init__(
        self,
        CONFIG,
    ):
        super(ChessTemporalTransformerEncoder, self).__init__()

        self.code = "EFT-Extended"

        # Existing configuration parameters
        self.vocab_sizes = CONFIG.VOCAB_SIZES
        self.d_model = CONFIG.D_MODEL
        self.n_heads = CONFIG.N_HEADS
        self.d_queries = CONFIG.D_QUERIES
        self.d_values = CONFIG.D_VALUES
        self.d_inner = CONFIG.D_INNER
        self.n_layers = CONFIG.N_LAYERS
        self.dropout = CONFIG.DROPOUT

        # Encoder remains the same
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

        # Prediction Heads
class ChessTemporalTransformerEncoder(nn.Module):
    """
    Extended Chess Transformer Encoder with additional prediction heads:
    1. From and To square prediction
    2. Game result prediction (white win/black win)
    3. Move time prediction
    """

    def __init__(
        self,
        CONFIG,
    ):
        super(ChessTemporalTransformerEncoder, self).__init__()

        self.code = "EFT-Extended"

        # Existing configuration parameters
        self.vocab_sizes = CONFIG.VOCAB_SIZES
        self.d_model = CONFIG.D_MODEL
        self.n_heads = CONFIG.N_HEADS
        self.d_queries = CONFIG.D_QUERIES
        self.d_values = CONFIG.D_VALUES
        self.d_inner = CONFIG.D_INNER
        self.n_layers = CONFIG.N_LAYERS
        self.dropout = CONFIG.DROPOUT

        # Encoder remains the same
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

        # Prediction Heads
        # 1. From/To square prediction heads (existing)
        self.from_squares = nn.Linear(self.d_model, 1)
        self.to_squares = nn.Linear(self.d_model, 1)

        # Pooling layers for global context
        self.game_result_pool = nn.Sequential(
            nn.Linear(self.d_model, 1),  # Attention weights
            nn.Softmax(dim=1)  # Normalize weights across sequence length
        )
        
        self.move_time_pool = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Softmax(dim=1)
        )
        
        self.game_length_pool = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Softmax(dim=1)
        )

        # 2. Game Result Prediction Head (outputs value between -1 and 1)
        self.game_result_head = nn.Sequential(
            # Initial dimensionality reduction
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            
            # Residual blocks
            ResidualBlock(self.d_model // 2, dropout=self.dropout),
            ResidualBlock(self.d_model // 2, dropout=self.dropout),
            ResidualBlock(self.d_model // 2, dropout=self.dropout),
            
            # Final layers
            nn.Linear(self.d_model // 2, self.d_model // 8),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            
            nn.Linear(self.d_model // 8, 1),
            nn.Tanh()  # Ensures output is between -1 and 1
        )

        # 3. Move Time Prediction Head (outputs value between 0 and 1)
        self.move_time_head = nn.Sequential(
            # Initial dimensionality reduction
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            
            # Residual blocks
            ResidualBlock(self.d_model // 2, dropout=self.dropout),
            ResidualBlock(self.d_model // 2, dropout=self.dropout),
            ResidualBlock(self.d_model // 2, dropout=self.dropout),
            
            # Final layers
            nn.Linear(self.d_model // 2, self.d_model // 8),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            
            nn.Linear(self.d_model // 8, 1),
            nn.Sigmoid()  # Ensures output is between 0 and 1
        )
        
        # 4. Predicts number of full moves left in the game
        self.game_length_head = nn.Sequential(
            # Initial dimensionality reduction
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            
            # Residual blocks
            ResidualBlock(self.d_model // 2, dropout=self.dropout),
            ResidualBlock(self.d_model // 2, dropout=self.dropout),
            ResidualBlock(self.d_model // 2, dropout=self.dropout),
            
            # Final layers
            nn.Linear(self.d_model // 2, self.d_model // 8),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            
            nn.Linear(self.d_model // 8, 1),
        )

        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """
        Initializes weights for all layers in the model.
        """
        def _init_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Embedding):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.constant_(layer.bias, 0)
                nn.init.constant_(layer.weight, 1.0)

        # Apply initialization to all submodules
        self.apply(_init_layer)

        # Specific initialization for heads
        for head in [self.from_squares, self.to_squares]:
            nn.init.xavier_uniform_(head.weight)
            nn.init.constant_(head.bias, 0)

        for head in [self.game_result_head, self.move_time_head, self.game_length_head]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, batch):
        """
        Forward propagation with additional prediction heads

        Returns:
            dict: Dictionary containing all predictions
        """
        # Encoder
        boards = self.board_encoder(
            batch["turns"],
            batch["white_kingside_castling_rights"],
            batch["white_queenside_castling_rights"],
            batch["black_kingside_castling_rights"],
            batch["black_queenside_castling_rights"],
            batch["board_positions"],
            batch["time_control"],
            batch["move_number"],
            batch["num_legal_moves"],
            batch["white_remaining_time"],
            batch["black_remaining_time"],
            batch["phase"],
            batch["white_rating"],
            batch["black_rating"],
            batch["white_material_value"],
            batch["black_material_value"],
            batch["material_difference"],
        )  # (N, BOARD_STATUS_LENGTH, d_model)

        # From/To square predictions (unchanged)
        from_squares = (
            self.from_squares(boards[:, 16:, :]).squeeze(2).unsqueeze(1)
        )  # (N, 1, 64)
        to_squares = (
            self.to_squares(boards[:, 16:, :]).squeeze(2).unsqueeze(1)
        )  # (N, 1, 64)

        # Apply attention pooling for each prediction head
        # Game result prediction
        game_weights = self.game_result_pool(boards).unsqueeze(-1)  # (N, seq_len, 1)
        game_context = (boards * game_weights).sum(dim=1)  # (N, d_model)
        game_result = self.game_result_head(game_context)  # (N, 1)

        # Move time prediction
        time_weights = self.move_time_pool(boards).unsqueeze(-1)  # (N, seq_len, 1)
        time_context = (boards * time_weights).sum(dim=1)  # (N, d_model)
        move_time = self.move_time_head(time_context)  # (N, 1)
        
        # Game length prediction
        length_weights = self.game_length_pool(boards).unsqueeze(-1)  # (N, seq_len, 1)
        length_context = (boards * length_weights).sum(dim=1)  # (N, d_model)
        moves_until_end = self.game_length_head(length_context)  # (N, 1)
        
        predictions = {
            'from_squares': from_squares,
            'to_squares': to_squares,
            'game_result': game_result,
            'move_time': move_time * 100,  # Scaled for data compatibility
            'moves_until_end': moves_until_end
        }

        return predictions


if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name)

    # Model
    model = CONFIG.MODEL(CONFIG).to(DEVICE)
    print(
        "There are %d learnable parameters in this model."
        % sum([p.numel() for p in model.parameters()])
    )