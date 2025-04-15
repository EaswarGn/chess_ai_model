import math
import torch
import argparse
from torch import nn
import sys

from configs import import_config
from modules_ddp import BoardEncoder, OGBoardEncoder
import torch.nn.functional as F
import numpy as np
    

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
        DEVICE
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

        self.board_encoder = OGBoardEncoder(
            DEVICE=DEVICE,
            vocab_sizes=self.vocab_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            dropout=self.dropout,
            #num_cls_tokens=self.num_cls_tokens,
        )
        
        self.from_squares = nn.Linear(CONFIG.D_MODEL, 1)
        self.to_squares = nn.Linear(CONFIG.D_MODEL, 1)
        self.game_result_head = None
        self.move_time_head = CONFIG.move_time_head
        self.game_length_head = CONFIG.game_length_head
        self.categorical_game_result_head = CONFIG.categorical_game_result_head
        
        # Create task-specific CLS tokens
        self.moves_remaining_cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.game_result_cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.time_suggestion_cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))


        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """
        Initializes all weights and biases in the model to zeros.
        """
        def _init_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Embedding):
                nn.init.constant_(layer.weight, 0)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.constant_(layer.bias, 0)
                nn.init.constant_(layer.weight, 0)

        # Apply zero initialization to all submodules
        self.apply(_init_layer)

        # Also zero initialize the special prediction heads
        for head in [self.from_squares, self.to_squares]:
            if head is not None:
                nn.init.constant_(head.weight, 0)
                nn.init.constant_(head.bias, 0)

        for head in [self.game_result_head, self.move_time_head, self.game_length_head]:
            if head is not None:
                for layer in head:
                    if isinstance(layer, nn.Linear):
                        nn.init.constant_(layer.weight, 0)
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
        time_control = torch.cat([batch["base_time"], batch["increment_time"]], dim=-1)
        
        # Encoder
        boards = self.board_encoder(
            batch["turn"],
            batch["white_kingside_castling_rights"],
            batch["white_queenside_castling_rights"],
            batch["black_kingside_castling_rights"],
            batch["black_queenside_castling_rights"],
            batch["board_position"],
        )  # (N, BOARD_STATUS_LENGTH, d_model)
        
        from_squares = (self.from_squares(boards[:, 5:, :]).squeeze(2).unsqueeze(1)) if self.from_squares is not None else None
        to_squares = (self.to_squares(boards[:, 5:, :]).squeeze(2).unsqueeze(1)) if self.to_squares is not None else None
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

class PonderingTimeModel(nn.Module):
    """
    Extended Chess Transformer Encoder with additional prediction heads:
    1. From and To square prediction
    2. Game result prediction (white win/black win)
    3. Move time prediction
    """

    def __init__(
        self,
        CONFIG,
        DEVICE
    ):
        super(PonderingTimeModel, self).__init__()

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
        
        self.num_cls_tokens = 1

        # Encoder remains the same
        """self.board_encoder = BoardEncoder(
            DEVICE=DEVICE,
            vocab_sizes=self.vocab_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            dropout=self.dropout,
            num_cls_tokens=self.num_cls_tokens
        )"""
        self.board_encoder = BoardEncoder(
            DEVICE=DEVICE,
            vocab_sizes=self.vocab_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            dropout=self.dropout,
            num_cls_tokens=self.num_cls_tokens,
        )
        
        
        self.from_squares = None
        self.to_squares = None
        self.game_result_head = None
        self.move_time_head = CONFIG.move_time_head
        self.game_length_head = None
        self.categorical_game_result_head = None
        
        # Create task-specific CLS tokens
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

        for head in [self.move_time_head]:
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
            self.time_suggestion_cls_token.expand(batch_size, 1, self.d_model)
        ], dim=1)
        time_control = torch.cat([batch["base_time"], batch["increment_time"]], dim=-1)
        
        # Encoder
        boards = self.board_encoder(
            batch["turn"],
            batch["white_kingside_castling_rights"],
            batch["white_queenside_castling_rights"],
            batch["black_kingside_castling_rights"],
            batch["black_queenside_castling_rights"],
            batch["board_position"],
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
            time_control,
            cls_tokens,
        )  # (N, BOARD_STATUS_LENGTH, d_model)
        
        move_time = self.move_time_head(boards[:, 0:1, :]).squeeze(-1)
         
        predictions = {
            'from_squares': None,
            'to_squares': None,
            'game_result': None,
            'move_time': move_time, 
            'moves_until_end': None,
            'categorical_game_result': None
        }

        return predictions


if __name__ == "__main__":
    # Get configuration
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name)
    CONFIG = CONFIG.CONFIG()

    # Model
    model = None
    if "time" in CONFIG.NAME:
        model = PonderingTimeModel(CONFIG=CONFIG, DEVICE=DEVICE)
    else:
        model = ChessTemporalTransformerEncoder(CONFIG=CONFIG, DEVICE=DEVICE)
    print(
        "There are %d learnable parameters in this model."
        % sum([p.numel() for p in model.parameters() if p.requires_grad])
    )
