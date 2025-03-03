import math
import torch
import argparse
from torch import nn
import sys

from configs import import_config
from modules import BoardEncoder, OGBoardEncoder

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

        self.code = CONFIG.NAME

        # Existing configuration parameters
        self.vocab_sizes = CONFIG.VOCAB_SIZES
        self.d_model = CONFIG.D_MODEL
        self.n_heads = CONFIG.N_HEADS
        self.d_queries = CONFIG.D_QUERIES
        self.d_values = CONFIG.D_VALUES
        self.d_inner = CONFIG.D_INNER
        self.n_layers = CONFIG.N_LAYERS
        self.dropout = CONFIG.DROPOUT
        
        self.num_cls_tokens = 3

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
            num_cls_tokens=self.num_cls_tokens
        )
        
        self.from_squares = nn.Linear(CONFIG.D_MODEL, 1)
        self.to_squares = nn.Linear(CONFIG.D_MODEL, 1)
        self.game_result_head = None
        self.move_time_head = CONFIG.OUTPUTS["move_time"]
        self.game_length_head = CONFIG.OUTPUTS["moves_until_end"]
        self.categorical_game_result_head = CONFIG.OUTPUTS["categorical_game_result"]
        
        # Create task-specific CLS tokens
        self.moves_remaining_cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.game_result_cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.time_suggestion_cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))


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
            if head is not None:  # Ensure the head is not None before initializing
                nn.init.xavier_uniform_(head.weight)
                nn.init.constant_(head.bias, 0)

        for head in [self.game_result_head, self.move_time_head, self.game_length_head]:
            if head is not None:  # Ensure the head is not None before iterating over its layers
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
        
        batch_size = batch["turn"].size(0)
        # Expand CLS tokens for the batch
        cls_tokens = torch.cat([
            self.moves_remaining_cls_token.expand(batch_size, 1, self.d_model),
            self.game_result_cls_token.expand(batch_size, 1, self.d_model),
            self.time_suggestion_cls_token.expand(batch_size, 1, self.d_model)
        ], dim=1)
        
        # Encoder
        boards = self.board_encoder(
            batch["turn"],
            batch["white_kingside_castling_rights"],
            batch["white_queenside_castling_rights"],
            batch["black_kingside_castling_rights"],
            batch["black_queenside_castling_rights"],
            batch["board_position"],
            batch["time_control"],
            batch["move_number"],
            batch["num_legal_moves"],
            batch["white_remaining_time"],
            batch["black_remaining_time"],
            batch["phase"],
            #batch["white_rating"],
            #batch["black_rating"],
            batch["white_material_value"],
            batch["black_material_value"],
            batch["material_difference"],
            cls_tokens,
        )  # (N, BOARD_STATUS_LENGTH, d_model)
        
        
        from_squares = (self.from_squares(boards[:, 14+self.num_cls_tokens:, :]).squeeze(2).unsqueeze(1)) if self.from_squares is not None else None
        to_squares = (self.to_squares(boards[:, 14+self.num_cls_tokens:, :]).squeeze(2).unsqueeze(1)) if self.to_squares is not None else None
        moves_until_end = self.game_length_head(boards[:, 0:1, :]).squeeze(-1) if self.game_length_head is not None else None
        game_result = self.game_result_head(boards[:, 1:2, :]).squeeze(-1) if self.game_result_head is not None else None
        move_time = self.move_time_head(boards[:, 2:3, :]).squeeze(-1) if self.move_time_head is not None else None
        categorical_game_result = self.categorical_game_result_head(boards[:, 1:2, :]).squeeze(-1).squeeze(1) if self.categorical_game_result_head is not None else None
        
        
        
        predictions = {
            'from_squares': from_squares,
            'to_squares': to_squares,
            'game_result': game_result,
            'move_time': move_time, #* 100,  # Scaled for data compatibility
            'moves_until_end': moves_until_end,
            'categorical_game_result': categorical_game_result
        }

        return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name)

    # Model
    model = ChessTemporalTransformerEncoder(CONFIG=CONFIG, DEVICE=DEVICE)
    print(
        "There are %d learnable parameters in this model."
        % sum([p.numel() for p in model.parameters()])
    )
    

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