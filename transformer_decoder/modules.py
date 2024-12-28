import math
import tensorflow as tf

# Replace torch.device with tf.device
def get_device():
    return "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    The Multi-Head Attention sublayer.
    Adapted from PyTorch version to TensorFlow.
    """
    
    def __init__(self, d_model, n_heads, d_queries, d_values, dropout, in_decoder=False):
        """
        Init.
        """
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_keys = d_queries
        self.in_decoder = in_decoder
        
        # Replace nn.Linear with Dense layers
        self.cast_queries = tf.keras.layers.Dense(n_heads * d_queries)
        self.cast_keys_values = tf.keras.layers.Dense(n_heads * (d_queries + d_values))
        self.cast_output = tf.keras.layers.Dense(d_model)
        
        # Replace nn.LayerNorm with LayerNormalization
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Replace nn.Dropout with Dropout
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, query_sequences, key_value_sequences, key_value_sequence_lengths, training=False):
        """
        Forward pass.
        """
        batch_size = tf.shape(query_sequences)[0]
        query_sequence_pad_length = tf.shape(query_sequences)[1]
        key_value_sequence_pad_length = tf.shape(key_value_sequences)[1]

        shapes_equal = tf.reduce_all(tf.equal(tf.shape(key_value_sequences), tf.shape(query_sequences)))
    
        if not shapes_equal:
            self_attention = False
        else:
            # Check for self-attention
            self_attention = tf.reduce_all(tf.equal(key_value_sequences, query_sequences))
        
        # Store input for residual connection
        input_to_add = query_sequences
        
        # Apply layer normalization
        query_sequences = self.layer_norm(query_sequences)
        if self_attention:
            key_value_sequences = self.layer_norm(key_value_sequences)
            
        # Project input sequences
        queries = self.cast_queries(query_sequences)
        keys_values = self.cast_keys_values(key_value_sequences)
        keys, values = tf.split(keys_values, [self.n_heads * self.d_keys, self.n_heads * self.d_values], axis=-1)
        
        # Reshape for multi-head attention
        def reshape_for_multihead(x, sequence_length):
            x = tf.reshape(x, [batch_size, sequence_length, self.n_heads, -1])
            return tf.transpose(x, [0, 2, 1, 3])
            
        queries = reshape_for_multihead(queries, query_sequence_pad_length)
        keys = reshape_for_multihead(keys, key_value_sequence_pad_length)
        values = reshape_for_multihead(values, key_value_sequence_pad_length)
        
        # Scaled dot-product attention
        attention_weights = tf.matmul(queries, keys, transpose_b=True)
        attention_weights = attention_weights / tf.math.sqrt(tf.cast(self.d_keys, tf.float32))
        
        # Masking
        # Create mask for padded positions
        #key_value_sequence_lengths = tf.expand_dims(key_value_sequence_lengths, axis=0)
        mask = tf.sequence_mask(key_value_sequence_lengths, key_value_sequence_pad_length)
        mask = tf.expand_dims(tf.expand_dims(mask, 1), 2)
        mask = tf.repeat(mask, repeats=self.n_heads, axis=1)
        mask = tf.repeat(mask, repeats=query_sequence_pad_length, axis=2)
        
        with tf.device('/CPU:0'):
            attention_weights = tf.where(mask, attention_weights, tf.float32.min)
        
        # Add causal mask for decoder self-attention
        if self.in_decoder and self_attention:
            #print("causal mask implemented")
            causal_mask = tf.linalg.band_part(tf.ones((query_sequence_pad_length, key_value_sequence_pad_length)), -1, 0)
            causal_mask = tf.expand_dims(tf.expand_dims(causal_mask, 0), 0)
            attention_weights = tf.where(tf.cast(causal_mask, tf.bool), attention_weights, tf.float32.min)
        
        # Apply softmax and dropout
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Apply attention to values
        sequences = tf.matmul(attention_weights, values)
        
        # Reshape back to original dimensions
        sequences = tf.transpose(sequences, [0, 2, 1, 3])
        sequences = tf.reshape(sequences, [batch_size, query_sequence_pad_length, -1])
        
        # Final projection and residual connection
        sequences = self.cast_output(sequences)
        sequences = self.dropout(sequences, training=training) + input_to_add
        
        return sequences

class PositionWiseFCNetwork(tf.keras.layers.Layer):
    """
    The Position-Wise Feed Forward Network sublayer.
    """
    
    def __init__(self, d_model, d_inner, dropout):
        super(PositionWiseFCNetwork, self).__init__()
        
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.fc1 = tf.keras.layers.Dense(d_inner)
        self.fc2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.activation = tf.keras.layers.ReLU()
        
    def call(self, sequences, training=False):
        input_to_add = sequences
        
        sequences = self.layer_norm(sequences)
        sequences = self.fc1(sequences)
        sequences = self.activation(sequences)
        sequences = self.dropout(sequences, training=training)
        sequences = self.fc2(sequences)
        sequences = self.dropout(sequences, training=training) + input_to_add
        
        return sequences

class BoardEncoder(tf.keras.Model):
    def __init__(self, vocab_sizes, d_model, n_heads, d_queries, d_values, d_inner, n_layers, dropout):
        super(BoardEncoder, self).__init__()
        self.d_model = d_model

        self.turn_embeddings = tf.keras.layers.Embedding(vocab_sizes["turn"], d_model)
        self.white_kingside_castling_rights_embeddings = tf.keras.layers.Embedding(vocab_sizes["white_kingside_castling_rights"], d_model)
        self.white_queenside_castling_rights_embeddings = tf.keras.layers.Embedding(vocab_sizes["white_queenside_castling_rights"], d_model)
        self.black_kingside_castling_rights_embeddings = tf.keras.layers.Embedding(vocab_sizes["black_kingside_castling_rights"], d_model)
        self.black_queenside_castling_rights_embeddings = tf.keras.layers.Embedding(vocab_sizes["black_queenside_castling_rights"], d_model)
        self.board_position_embeddings = tf.keras.layers.Embedding(vocab_sizes["board_position"], d_model)

        self.positional_embeddings = tf.keras.layers.Embedding(69, d_model)

        self.encoder_layers = [self.make_encoder_layer(d_model, n_heads, d_queries, d_values, d_inner, dropout) for _ in range(n_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def make_encoder_layer(self, d_model, n_heads, d_queries, d_values, d_inner, dropout):
        return [
            MultiHeadAttention(d_model, n_heads, d_queries, d_values, dropout, in_decoder=False),
            PositionWiseFCNetwork(d_model, d_inner, dropout)
        ]

    def call(self, turns, white_kingside_castling_rights, white_queenside_castling_rights, black_kingside_castling_rights, black_queenside_castling_rights, board_positions, training=False):
        batch_size = tf.keras.backend.shape(turns)[0]

        embeddings = tf.concat([
            self.turn_embeddings(turns),
            self.white_kingside_castling_rights_embeddings(white_kingside_castling_rights),
            self.white_queenside_castling_rights_embeddings(white_queenside_castling_rights),
            self.black_kingside_castling_rights_embeddings(black_kingside_castling_rights),
            self.black_queenside_castling_rights_embeddings(black_queenside_castling_rights),
            self.board_position_embeddings(board_positions)
        ], axis=1)

        positions = tf.range(tf.keras.backend.shape(embeddings)[1])
        boards = embeddings + self.positional_embeddings(positions)
        boards = boards * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        boards = self.dropout(boards, training=training)
        for encoder_layer in self.encoder_layers:
            key_value_sequence_lengths = tf.fill([batch_size], 69)
            boards = encoder_layer[0](boards, boards, key_value_sequence_lengths, training=training)
            boards = encoder_layer[1](boards, training=training)

        boards = self.layer_norm(boards)

        return boards

class MoveDecoder(tf.keras.Model):
    """
    The Move Decoder.
    """
    
    def __init__(self, vocab_size, n_moves, d_model, n_heads, d_queries, d_values, d_inner, n_layers, dropout):
        super(MoveDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.n_moves = n_moves
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.d_model = d_model
        
        # Embedding layers
        self.embeddings = tf.keras.layers.Embedding(vocab_size, d_model)
        self.positional_embeddings = tf.keras.layers.Embedding(n_moves, d_model)
        
        # Create decoder layers
        self.decoder_layers = [
            [
                MultiHeadAttention(d_model, n_heads, d_queries, d_values, dropout, in_decoder=True),
                MultiHeadAttention(d_model, n_heads, d_queries, d_values, dropout, in_decoder=True),
                PositionWiseFCNetwork(d_model, d_inner, dropout)
            ]
            for _ in range(n_layers)
        ]
        
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
    def call(self, moves, lengths, boards, training=False):
        # Compute embeddings
        embeddings = self.embeddings(moves)
        
        if tf.shape(lengths)[0] == 1:
            lengths = tf.squeeze(lengths, axis=0)
        
        # Add positional embeddings
        moves = embeddings + self.positional_embeddings(tf.range(self.n_moves))[tf.newaxis, :]  # (N, n_moves, d_model)
        moves *= math.sqrt(self.d_model)  # (N, n_moves, d_model)
        
        # Apply dropout
        moves_embedded = self.dropout(moves, training=training)
        
        # Process through decoder layers
        for self_attention, cross_attention, pwfcn in self.decoder_layers:
            if tf.shape(lengths)[0] > 1:
                lengths = tf.squeeze(lengths)
            moves_embedded = self_attention(
                moves_embedded, moves_embedded, lengths,
                training=training
            )
            moves_embedded = cross_attention(
                moves_embedded, boards,
                tf.ones([tf.shape(boards)[0]], dtype=tf.int32) * 69,
                training=training
            )
            moves_embedded = pwfcn(moves_embedded, training=training)
            
        # Apply final layer normalization and projection
        moves_embedded = self.layer_norm(moves_embedded)
        output = self.fc(moves_embedded)
        
        return output