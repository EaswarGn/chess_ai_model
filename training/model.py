"""
Enhanced Chess Transformer with Comprehensive Temporal Modeling
"""
import torch
import torch.nn as nn
import math
from modules import OGBoardEncoder, MoveDecoder

import math

def init_weights(module):
    """
    Initialize weights using Xavier/Glorot uniform initialization.
    Applies to Linear, Embedding, and Conv layers.
    
    Args:
        module (nn.Module): The module to initialize weights for
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Embedding):
        nn.init.uniform_(
            module.weight, 
            a=-1.0 / math.sqrt(module.embedding_dim), 
            b=1.0 / math.sqrt(module.embedding_dim)
        )
    
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

class TimeControlEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Vocabulary-based time control embedding
        self.time_control_embedding = nn.Embedding(
            num_embeddings=457,  # Unique time control combinations
            embedding_dim=config.D_MODEL // 4  # Proportional to model dimension
        )
        
        # Time remaining encoder
        self.time_remaining_encoder = nn.Sequential(
            nn.Linear(2, config.D_MODEL // 4),  # White and black remaining time
            nn.ReLU(),
            nn.Linear(config.D_MODEL // 4, config.D_MODEL // 4)
        )
        
        # Time pressure detection module
        self.time_pressure_detector = nn.Sequential(
            nn.Linear(2, config.D_MODEL // 8),
            nn.ReLU(),
            nn.Linear(config.D_MODEL // 8, config.D_MODEL // 8),
            nn.Sigmoid()  # Outputs time pressure intensity
        )

    def forward(self, time_control_id, white_remaining, black_remaining):
        """
        Args:
            time_control_id (torch.Tensor): Unique identifier for time control
            white_remaining (torch.Tensor): White player's remaining time
            black_remaining (torch.Tensor): Black player's remaining time
        
        Returns:
            torch.Tensor: Comprehensive time control embedding
        """
        # Embed time control
        time_control_embed = self.time_control_embedding(time_control_id)
        
        # Encode remaining time
        remaining_time_embed = self.time_remaining_encoder(
            torch.stack([white_remaining, black_remaining], dim=-1)
        )
        
        # Detect time pressure
        time_pressure = self.time_pressure_detector(
            torch.stack([white_remaining, black_remaining], dim=-1)
        )
        
        # Combine embeddings
        combined_time_embed = torch.cat([
            time_control_embed, 
            remaining_time_embed, 
            time_pressure
        ], dim=-1)
        
        return combined_time_embed

class PlayerCharacteristicsEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Player rating embedding
        self.rating_embedding = nn.Sequential(
            nn.Linear(2, config.D_MODEL // 4),
            nn.ReLU(),
            nn.Linear(config.D_MODEL // 4, config.D_MODEL // 4)
        )
        
        # Game phase embedding
        self.game_phase_embedding = nn.Embedding(
            num_embeddings=3,  # 0-3 game phases
            embedding_dim=config.D_MODEL // 8
        )

    def forward(self, white_rating, black_rating, game_phase):
        # Rating embeddings
        rating_embedding = self.rating_embedding(
            torch.stack([white_rating, black_rating], dim=-1)
        )
        
        # Game phase embedding
        phase_embedding = self.game_phase_embedding(game_phase)
        
        # Combine embeddings
        combined_player_embed = torch.cat([
            rating_embedding, 
            phase_embedding
        ], dim=-1)
        
        return combined_player_embed

class EnhancedChessTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Core transformer components
        self.board_encoder = OGBoardEncoder(
            vocab_sizes=config.VOCAB_SIZES,
            d_model=config.D_MODEL,
            n_heads=config.N_HEADS,
            d_queries=config.D_QUERIES,
            d_values=config.D_VALUES,
            d_inner=config.D_INNER,
            n_layers=config.N_LAYERS,
            dropout=config.DROPOUT
        )
        
        self.move_decoder = MoveDecoder(
            vocab_size=config.VOCAB_SIZES['moves'],
            n_moves=config.N_MOVES,
            d_model=config.D_MODEL,
            n_heads=config.N_HEADS,
            d_queries=config.D_QUERIES,
            d_values=config.D_VALUES,
            d_inner=config.D_INNER,
            n_layers=config.N_LAYERS,
            dropout=config.DROPOUT
        )
        
        # Temporal and player feature embeddings
        self.time_control_embedding = TimeControlEmbedding(config)
        self.player_characteristics_embedding = PlayerCharacteristicsEmbedding(config)
        
        # Multi-task prediction heads
        self.move_prediction_head = nn.Linear(config.D_MODEL, config.VOCAB_SIZES['moves'])
        
        self.time_spent_head = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_MODEL // 2),
            nn.ReLU(),
            nn.Linear(config.D_MODEL // 2, 1),  # Predicts time spent on move
        )
        
        self.game_outcome_head = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_MODEL // 2),
            nn.ReLU(),
            nn.Linear(config.D_MODEL // 2, 1),
            nn.Tanh()  # Ensures output is between -1 and 1
        )
        
        # Feature fusion layer
        self.feature_fusion = nn.Linear(
            config.D_MODEL + (config.D_MODEL // 4 * 2),  # Board + Time + Player features
            config.D_MODEL
        )
        
        self.apply(init_weights)

    def forward(self, batch):
        # Extract and embed temporal features
        time_control_embed = self.time_control_embedding(
            batch['time_control_id'], 
            batch['white_remaining_time'], 
            batch['black_remaining_time']
        )
        
        player_characteristics_embed = self.player_characteristics_embedding(
            batch['white_rating'], 
            batch['black_rating'], 
            batch['phase']
        )
        
        # Encode board state
        board_encoding = self.board_encoder(
            batch['turns'],
            batch['white_kingside_castling_rights'],
            batch['white_queenside_castling_rights'],
            batch['black_kingside_castling_rights'],
            batch['black_queenside_castling_rights'],
            batch['board_positions']
        )
        
        # Fuse all features
        fused_encoding = self.feature_fusion(
            torch.cat([
                board_encoding.mean(dim=1),  # Aggregate board encoding
                time_control_embed,
                player_characteristics_embed
            ], dim=-1)
        )
        
        # Prepare for decoder
        expanded_fused_encoding = fused_encoding.unsqueeze(1).expand(
            -1, board_encoding.size(1), -1
        )
        
        # Combine fused features with original board encoding
        combined_encoding = board_encoding + expanded_fused_encoding
        
        # Decode moves
        move_predictions = self.move_decoder(
            batch['moves'][:, :-1], 
            batch['lengths'].squeeze(1), 
            combined_encoding
        )
        
        # Multi-task predictions
        outputs = {
            'move_probabilities': move_predictions
        }
        
        # Additional prediction heads
        time_spent_prediction = self.time_spent_head(fused_encoding)
        game_outcome_prediction = self.game_outcome_head(fused_encoding)
        
        outputs['time_spent'] = time_spent_prediction
        outputs['game_outcome'] = game_outcome_prediction
        
        return outputs