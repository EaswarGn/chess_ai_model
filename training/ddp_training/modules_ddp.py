import math
import torch
from torch import nn
import sys


class MultiHeadAttention(nn.Module):
    """
    The Multi-Head Attention sublayer.

    Reused from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation.
    """

    def __init__(
        self, DEVICE, d_model, n_heads, d_queries, d_values, dropout, in_decoder=False
    ):
        """
        Init.

        Args:

            d_model (int): The size of vectors throughout the
            transformer model, i.e. input and output sizes for this
            sublayer.

            n_heads (int): The number of heads in the multi-head
            attention.

            d_queries (int): The size of query vectors (and also the
            size of the key vectors).

            d_values (int): The size of value vectors.

            dropout (float): The dropout probability.

            in_decoder (bool, optional): Is this Multi-Head Attention
            sublayer instance in the Decoder? Defaults to False.
        """
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_queries = d_queries
        self.d_values = d_values
        self.d_keys = d_queries  # size of key vectors, same as of the query vectors to allow dot-products for similarity

        self.in_decoder = in_decoder

        # A linear projection to cast (n_heads sets of) queries from the
        # input query sequences
        self.cast_queries = nn.Linear(d_model, n_heads * d_queries)

        # A linear projection to cast (n_heads sets of) keys and values
        # from the input reference sequences
        self.cast_keys_values = nn.Linear(d_model, n_heads * (d_queries + d_values))

        # A linear projection to cast (n_heads sets of) computed
        # attention-weighted vectors to output vectors (of the same size
        # as input query vectors)
        self.cast_output = nn.Linear(n_heads * d_values, d_model)

        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)
        
        self.DEVICE = DEVICE

    def forward(self, query_sequences, key_value_sequences, key_value_sequence_lengths):
        """
        Forward prop.

        Args:

            query_sequences (torch.FloatTensor): The input query
            sequences, of size (N, query_sequence_pad_length, d_model).

            key_value_sequences (torch.FloatTensor): The sequences to be
            queried against, of size (N, key_value_sequence_pad_length,
            d_model).

            key_value_sequence_lengths (torch.LongTensor): The true
            lengths of the key_value_sequences, to be able to ignore
            pads, of size (N).

        Returns:

            torch.FloatTensor: Attention-weighted output sequences for
            the query sequences, of size (N, query_sequence_pad_length,
            d_model).
        """
        batch_size = query_sequences.size(0)  # batch size (N) in number of sequences
        query_sequence_pad_length = query_sequences.size(1)
        key_value_sequence_pad_length = key_value_sequences.size(1)

        # Is this self-attention?
        self_attention = torch.equal(key_value_sequences, query_sequences)

        # Store input for adding later
        input_to_add = query_sequences.clone()

        # Apply layer normalization
        query_sequences = self.layer_norm(
            query_sequences
        )  # (N, query_sequence_pad_length, d_model)
        # If this is self-attention, do the same for the key-value
        # sequences (as they are the same as the query sequences) If
        # this isn't self-attention, they will already have been normed
        # in the last layer of the Encoder (from whence they came)
        if self_attention:
            key_value_sequences = self.layer_norm(
                key_value_sequences
            )  # (N, key_value_sequence_pad_length, d_model)

        # Project input sequences to queries, keys, values
        queries = self.cast_queries(
            query_sequences
        )  # (N, query_sequence_pad_length, n_heads * d_queries)
        keys, values = self.cast_keys_values(key_value_sequences).split(
            split_size=self.n_heads * self.d_keys, dim=-1
        )  # (N, key_value_sequence_pad_length, n_heads * d_keys), (N, key_value_sequence_pad_length, n_heads * d_values)

        # Split the last dimension by the n_heads subspaces
        queries = queries.contiguous().view(
            batch_size, query_sequence_pad_length, self.n_heads, self.d_queries
        )  # (N, query_sequence_pad_length, n_heads, d_queries)
        keys = keys.contiguous().view(
            batch_size, key_value_sequence_pad_length, self.n_heads, self.d_keys
        )  # (N, key_value_sequence_pad_length, n_heads, d_keys)
        values = values.contiguous().view(
            batch_size, key_value_sequence_pad_length, self.n_heads, self.d_values
        )  # (N, key_value_sequence_pad_length, n_heads, d_values)

        # Re-arrange axes such that the last two dimensions are the
        # sequence lengths and the queries/keys/values And then, for
        # convenience, convert to 3D tensors by merging the batch and
        # n_heads dimensions This is to prepare it for the batch matrix
        # multiplication (i.e. the dot product)
        queries = (
            queries.permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, query_sequence_pad_length, self.d_queries)
        )  # (N * n_heads, query_sequence_pad_length, d_queries)
        keys = (
            keys.permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, key_value_sequence_pad_length, self.d_keys)
        )  # (N * n_heads, key_value_sequence_pad_length, d_keys)
        values = (
            values.permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, key_value_sequence_pad_length, self.d_values)
        )  # (N * n_heads, key_value_sequence_pad_length, d_values)

        # Perform multi-head attention

        # Perform dot-products
        attention_weights = torch.bmm(
            queries, keys.permute(0, 2, 1)
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Scale dot-products
        attention_weights = (
            1.0 / math.sqrt(self.d_keys)
        ) * attention_weights  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Before computing softmax weights, prevent queries from
        # attending to certain keys

        # MASK 1: keys that are pads
        not_pad_in_keys = (
            torch.LongTensor(range(key_value_sequence_pad_length))
            .unsqueeze(0)
            .unsqueeze(0)
            .expand_as(attention_weights)
            .to(self.DEVICE)
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        not_pad_in_keys = (
            not_pad_in_keys
            < key_value_sequence_lengths.repeat_interleave(self.n_heads)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand_as(attention_weights)
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        # Note: PyTorch auto-broadcasts singleton dimensions in
        # comparison operations (as well as arithmetic operations)

        # Mask away by setting such weights to a large negative number,
        # so that they evaluate to 0 under the softmax
        attention_weights = attention_weights.masked_fill(
            ~not_pad_in_keys, -float("inf")
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        not_future_mask = ""
        # MASK 2: if this is self-attention in the Decoder, keys
        # chronologically ahead of queries
        if self.in_decoder and self_attention:
            # Therefore, a position [n, i, j] is valid only if j <= i
            # torch.tril(), i.e. lower triangle in a 2D matrix, sets j >
            # i to 0
            not_future_mask = (
                torch.ones_like(attention_weights).tril().bool().to(self.DEVICE)
            )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

            # Mask away by setting such weights to a large negative
            # number, so that they evaluate to 0 under the softmax
            attention_weights = attention_weights.masked_fill(
                ~not_future_mask, -float("inf")
            )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Compute softmax along the key dimension
        attention_weights = self.softmax(
            attention_weights
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Apply dropout
        attention_weights = self.apply_dropout(
            attention_weights
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Calculate sequences as the weighted sums of values based on
        # these softmax weights
        sequences = torch.bmm(
            attention_weights, values
        )  # (N * n_heads, query_sequence_pad_length, d_values)

        # Unmerge batch and n_heads dimensions and restore original
        # order of axes
        sequences = (
            sequences.contiguous()
            .view(batch_size, self.n_heads, query_sequence_pad_length, self.d_values)
            .permute(0, 2, 1, 3)
        )  # (N, query_sequence_pad_length, n_heads, d_values)

        # Concatenate the n_heads subspaces (each with an output of size
        # d_values)
        sequences = sequences.contiguous().view(
            batch_size, query_sequence_pad_length, -1
        )  # (N, query_sequence_pad_length, n_heads * d_values)

        # Transform the concatenated subspace-sequences into a single
        # output of size d_model
        sequences = self.cast_output(
            sequences
        )  # (N, query_sequence_pad_length, d_model)

        # Apply dropout and residual connection
        sequences = (
            self.apply_dropout(sequences) + input_to_add
        )  # (N, query_sequence_pad_length, d_model)
        
        del not_pad_in_keys
        del not_future_mask
        
        torch.cuda.empty_cache()

        return sequences


class PositionWiseFCNetwork(nn.Module):
    """
    The Position-Wise Feed Forward Network sublayer.

    Reused from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation.
    """

    def __init__(self, d_model, d_inner, dropout):
        """
        Init.

        Args:

            d_model (int): The size of vectors throughout the
            transformer model, i.e. input and output sizes for this
            sublayer.

            d_inner (int): An intermediate size.

            dropout (float): The dropout probability.
        """
        super(PositionWiseFCNetwork, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

        # A linear layer to project from the input size to an
        # intermediate size
        self.fc1 = nn.Linear(d_model, d_inner)

        # ReLU
        self.relu = nn.ReLU()

        # A linear layer to project from the intermediate size to the
        # output size (same as the input size)
        self.fc2 = nn.Linear(d_inner, d_model)

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, sequences):
        """
        Forward prop.

        Args:

            sequences (torch.FloatTensor): The input sequences, of size
            (N, pad_length, d_model).

        Returns:

            torch.FloatTensor: The transformed output sequences, of size
            (N, pad_length, d_model).
        """

        # Store input for adding later
        input_to_add = sequences.clone()  # (N, pad_length, d_model)

        # Apply layer-norm
        sequences = self.layer_norm(sequences)  # (N, pad_length, d_model)

        # Transform position-wise
        sequences = self.apply_dropout(
            self.relu(self.fc1(sequences))
        )  # (N, pad_length, d_inner)
        sequences = self.fc2(sequences)  # (N, pad_length, d_model)

        # Apply dropout and residual connection
        sequences = (
            self.apply_dropout(sequences) + input_to_add
        )  # (N, pad_length, d_model)

        return sequences
    

class BoardEncoder(nn.Module):
    """
    Enhanced Board Encoder with additional temporal and contextual features.
    """

    def __init__(
        self,
        DEVICE,
        vocab_sizes,
        d_model,
        n_heads,
        d_queries,
        d_values,
        d_inner,
        n_layers,
        dropout,
        num_cls_tokens
    ):
        """
        Initialize the Enhanced Board Encoder.

        Args:
            vocab_sizes (dict): Vocabulary sizes for various features.
            d_model (int): Dimension of the model's vector space.
            n_heads (int): Number of attention heads.
            d_queries (int): Dimension of query vectors.
            d_values (int): Dimension of value vectors.
            d_inner (int): Inner dimension of feed-forward networks.
            n_layers (int): Number of encoder layers.
            dropout (float): Dropout probability.
        """
        super(BoardEncoder, self).__init__()

        # Store configuration
        self.vocab_sizes = vocab_sizes
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout

        self.DEVICE = DEVICE

        # Existing Embeddings
        self.turn_embeddings = nn.Embedding(vocab_sizes["turn"], d_model, dtype=torch.float)
        self.white_kingside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["white_kingside_castling_rights"], d_model, dtype=torch.float
        )
        self.white_queenside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["white_queenside_castling_rights"], d_model, dtype=torch.float
        )
        self.black_kingside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["black_kingside_castling_rights"], d_model, dtype=torch.float
        )
        self.black_queenside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["black_queenside_castling_rights"], d_model, dtype=torch.float
        )
        self.board_position_embeddings = nn.Embedding(
            vocab_sizes["board_position"], d_model, dtype=torch.float
        )
        self.seq_length = 78 + num_cls_tokens
        self.positional_embeddings = nn.Embedding(self.seq_length, d_model, dtype=torch.float)

        """# New Temporal and Contextual Embeddings
        self.time_control_embeddings = nn.Embedding(
            vocab_sizes.get("time_control", 458), d_model, dtype=torch.float
        )"""
        
        
        
        # Continuous Feature Projections (ensure output remains float)
        self.time_control_projection = nn.Sequential(
            nn.Linear(2, d_model),  # Input size is 2 (initial time + increment time)
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        self.move_number_projection = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        self.num_legal_moves_projection = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        self.white_remaining_time_projection = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        self.black_remaining_time_projection = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Phase and Rating Embeddings
        self.phase_embeddings = nn.Embedding(
            vocab_sizes.get("phase", 3), d_model, dtype=torch.float
        )
        self.white_rating_embeddings = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.black_rating_embeddings = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        self.white_material_value_embeddings = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        self.black_material_value_embeddings = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        self.material_difference_embeddings = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Temporal Feature Fusion Layer
        self.temporal_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [self.make_encoder_layer() for _ in range(n_layers)]
        )

        # Dropout and Layer Norm
        self.apply_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def make_encoder_layer(self):
        """
        Create a single encoder layer with multi-head attention and 
        position-wise feed-forward network.
        """
        return nn.ModuleList([
            MultiHeadAttention(
                DEVICE=self.DEVICE,
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_queries=self.d_queries,
                d_values=self.d_values,
                dropout=self.dropout,
                in_decoder=False,
            ),
            PositionWiseFCNetwork(
                d_model=self.d_model, 
                d_inner=self.d_inner, 
                dropout=self.dropout
            ),
        ])

    def forward(
        self,
        turns,
        white_kingside_castling_rights,
        white_queenside_castling_rights,
        black_kingside_castling_rights,
        black_queenside_castling_rights,
        board_positions,
        # New input parameters for temporal features
        categorical_time_control,
        move_number,
        num_legal_moves,
        white_remaining_time,
        black_remaining_time,
        phase,
        #white_rating,
        #black_rating,
        white_material_value,
        black_material_value,
        material_difference,
        time_control,
        cls_tokens
    ):
        """
        Forward pass with enhanced temporal feature integration.
        
        Args:
            (Previous args remain the same)
            time_control (torch.LongTensor): Combined time control index
            move_number (torch.FloatTensor): Current move number
            num_legal_moves (torch.FloatTensor): Number of legal moves
            white_remaining_time (torch.FloatTensor): Remaining time for white
            black_remaining_time (torch.FloatTensor): Remaining time for black
            phase (torch.LongTensor): Game phase
            white_rating (torch.LongTensor): White player's rating
            black_rating (torch.LongTensor): Black player's rating
        
        Returns:
            torch.FloatTensor: Encoded board representation
        """
        batch_size = turns.size(0)
        
        # Ensure all tensors have the same dtype, e.g., float32
        embeddings = torch.cat(
            [
                # New features (ensure they are float as well)
                #Input to linear layers must be float dtype
                self.move_number_projection(move_number.unsqueeze(-1).to(torch.float32)),
                self.num_legal_moves_projection(num_legal_moves.unsqueeze(-1).to(torch.float32)),
                self.white_remaining_time_projection(white_remaining_time.unsqueeze(-1).to(torch.float32)),
                self.black_remaining_time_projection(black_remaining_time.unsqueeze(-1).to(torch.float32)),
                self.time_control_projection(time_control.to(torch.float32)).unsqueeze(1).to(torch.float32),
                self.phase_embeddings(phase),
                #self.white_rating_embeddings(white_rating.unsqueeze(-1).to(torch.float32)),
                #self.black_rating_embeddings(black_rating.unsqueeze(-1).to(torch.float32)),
                self.white_material_value_embeddings(white_material_value.unsqueeze(-1).to(torch.float32)),
                self.black_material_value_embeddings(black_material_value.unsqueeze(-1).to(torch.float32)),
                self.material_difference_embeddings(material_difference.unsqueeze(-1).to(torch.float32)),

                self.turn_embeddings(turns).to(torch.float32),  # Ensure embeddings are float
                self.white_kingside_castling_rights_embeddings(white_kingside_castling_rights).to(torch.float32),
                self.white_queenside_castling_rights_embeddings(white_queenside_castling_rights).to(torch.float32),
                self.black_kingside_castling_rights_embeddings(black_kingside_castling_rights).to(torch.float32),
                self.black_queenside_castling_rights_embeddings(black_queenside_castling_rights).to(torch.float32),
                self.board_position_embeddings(board_positions).to(torch.float32),
                
                
            ],
            dim=1
        )
        
        # Prepend CLS tokens
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)

        # Positional embeddings and scaling
        boards = embeddings + self.positional_embeddings.weight.unsqueeze(0)
        boards = boards * math.sqrt(self.d_model)

        # Dropout
        boards = self.apply_dropout(boards)
        
        seq_length = self.seq_length

        # Encoder layers
        for encoder_layer in self.encoder_layers:
            boards = encoder_layer[0](
                query_sequences=boards,
                key_value_sequences=boards,
                key_value_sequence_lengths=torch.LongTensor([seq_length] * batch_size).to(boards.device),
            )
            boards = encoder_layer[1](sequences=boards)

        # Layer normalization
        boards = self.layer_norm(boards)

        return boards


#seeing if adding batch normalization affects predictive performance
class ExperimentalBoardEncoder(nn.Module):
    """
    Enhanced Board Encoder with additional temporal and contextual features.
    """

    def __init__(
        self,
        DEVICE,
        vocab_sizes,
        d_model,
        n_heads,
        d_queries,
        d_values,
        d_inner,
        n_layers,
        dropout,
        num_cls_tokens
    ):
        """
        Initialize the Enhanced Board Encoder.

        Args:
            vocab_sizes (dict): Vocabulary sizes for various features.
            d_model (int): Dimension of the model's vector space.
            n_heads (int): Number of attention heads.
            d_queries (int): Dimension of query vectors.
            d_values (int): Dimension of value vectors.
            d_inner (int): Inner dimension of feed-forward networks.
            n_layers (int): Number of encoder layers.
            dropout (float): Dropout probability.
        """
        super(ExperimentalBoardEncoder, self).__init__()

        # Store configuration
        self.vocab_sizes = vocab_sizes
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout

        self.DEVICE = DEVICE

        # Existing Embeddings
        self.turn_embeddings = nn.Embedding(vocab_sizes["turn"], d_model, dtype=torch.float)
        self.white_kingside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["white_kingside_castling_rights"], d_model, dtype=torch.float
        )
        self.white_queenside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["white_queenside_castling_rights"], d_model, dtype=torch.float
        )
        self.black_kingside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["black_kingside_castling_rights"], d_model, dtype=torch.float
        )
        self.black_queenside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["black_queenside_castling_rights"], d_model, dtype=torch.float
        )
        self.board_position_embeddings = nn.Embedding(
            vocab_sizes["board_position"], d_model, dtype=torch.float
        )
        self.seq_length = 78 + num_cls_tokens
        self.positional_embeddings = nn.Embedding(self.seq_length, d_model, dtype=torch.float)

        """# New Temporal and Contextual Embeddings
        self.time_control_embeddings = nn.Embedding(
            vocab_sizes.get("time_control", 458), d_model, dtype=torch.float
        )"""
        
        
        
        # Continuous Feature Projections (ensure output remains float)
        self.time_control_projection = nn.Sequential(
            nn.Linear(2, d_model),  # Input size is 2 (initial time + increment time)
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        self.move_number_projection = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        self.num_legal_moves_projection = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        self.white_remaining_time_projection = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        self.black_remaining_time_projection = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Phase and Rating Embeddings
        self.phase_embeddings = nn.Embedding(
            vocab_sizes.get("phase", 3), d_model, dtype=torch.float
        )
        self.white_rating_embeddings = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.black_rating_embeddings = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        self.white_material_value_embeddings = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        self.black_material_value_embeddings = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        self.material_difference_embeddings = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Temporal Feature Fusion Layer
        self.temporal_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [self.make_encoder_layer() for _ in range(n_layers)]
        )

        # Dropout and Layer Norm
        self.apply_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        num_features = 9
        self.batch_norm_layers = []
        for _ in range(num_features):
            self.batch_norm_layers.append(nn.BatchNorm1d(1))

    def make_encoder_layer(self):
        """
        Create a single encoder layer with multi-head attention and 
        position-wise feed-forward network.
        """
        return nn.ModuleList([
            MultiHeadAttention(
                DEVICE=self.DEVICE,
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_queries=self.d_queries,
                d_values=self.d_values,
                dropout=self.dropout,
                in_decoder=False,
            ),
            PositionWiseFCNetwork(
                d_model=self.d_model, 
                d_inner=self.d_inner, 
                dropout=self.dropout
            ),
        ])

    def forward(
        self,
        turns,
        white_kingside_castling_rights,
        white_queenside_castling_rights,
        black_kingside_castling_rights,
        black_queenside_castling_rights,
        board_positions,
        # New input parameters for temporal features
        categorical_time_control,
        move_number,
        num_legal_moves,
        white_remaining_time,
        black_remaining_time,
        phase,
        #white_rating,
        #black_rating,
        white_material_value,
        black_material_value,
        material_difference,
        time_control,
        cls_tokens
    ):
        """
        Forward pass with enhanced temporal feature integration.
        
        Args:
            (Previous args remain the same)
            time_control (torch.LongTensor): Combined time control index
            move_number (torch.FloatTensor): Current move number
            num_legal_moves (torch.FloatTensor): Number of legal moves
            white_remaining_time (torch.FloatTensor): Remaining time for white
            black_remaining_time (torch.FloatTensor): Remaining time for black
            phase (torch.LongTensor): Game phase
            white_rating (torch.LongTensor): White player's rating
            black_rating (torch.LongTensor): Black player's rating
        
        Returns:
            torch.FloatTensor: Encoded board representation
        """
        batch_size = turns.size(0)
        
        #print(move_number.shape)
        #print(time_control.shape)
        #move_number = self.batch_norm_layers[0](move_number)


        # Ensure all tensors have the same dtype, e.g., float32
        embeddings = torch.cat(
            [
                # New features (ensure they are float as well)
                #Input to linear layers must be float dtype
                self.move_number_projection(move_number.unsqueeze(-1).to(torch.float32)),
                self.num_legal_moves_projection(num_legal_moves.unsqueeze(-1).to(torch.float32)),
                self.white_remaining_time_projection(white_remaining_time.unsqueeze(-1).to(torch.float32)),
                self.black_remaining_time_projection(black_remaining_time.unsqueeze(-1).to(torch.float32)),
                self.time_control_projection(time_control.to(torch.float32)).unsqueeze(1).to(torch.float32),
                self.phase_embeddings(phase),
                #self.white_rating_embeddings(white_rating.unsqueeze(-1).to(torch.float32)),
                #self.black_rating_embeddings(black_rating.unsqueeze(-1).to(torch.float32)),
                self.white_material_value_embeddings(white_material_value.unsqueeze(-1).to(torch.float32)),
                self.black_material_value_embeddings(black_material_value.unsqueeze(-1).to(torch.float32)),
                self.material_difference_embeddings(material_difference.unsqueeze(-1).to(torch.float32)),

                self.turn_embeddings(turns).to(torch.float32),  # Ensure embeddings are float
                self.white_kingside_castling_rights_embeddings(white_kingside_castling_rights).to(torch.float32),
                self.white_queenside_castling_rights_embeddings(white_queenside_castling_rights).to(torch.float32),
                self.black_kingside_castling_rights_embeddings(black_kingside_castling_rights).to(torch.float32),
                self.black_queenside_castling_rights_embeddings(black_queenside_castling_rights).to(torch.float32),
                self.board_position_embeddings(board_positions).to(torch.float32),
                
                
            ],
            dim=1
        )
        
        # Prepend CLS tokens
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)

        # Positional embeddings and scaling
        boards = embeddings + self.positional_embeddings.weight.unsqueeze(0)
        boards = boards * math.sqrt(self.d_model)

        # Dropout
        boards = self.apply_dropout(boards)
        
        seq_length = self.seq_length

        # Encoder layers
        for encoder_layer in self.encoder_layers:
            boards = encoder_layer[0](
                query_sequences=boards,
                key_value_sequences=boards,
                key_value_sequence_lengths=torch.LongTensor([seq_length] * batch_size).to(boards.device),
            )
            boards = encoder_layer[1](sequences=boards)

        # Layer normalization
        boards = self.layer_norm(boards)

        return boards