import os
import h5py
import tensorflow as tf
import tables as tb

class ChessDataset:
    def __init__(self, data_folder, h5_file, n_moves=None, **unused):
        """
        Args:
            data_folder (str): Path to the data folder containing the H5 file.
            h5_file (str): Name of the H5 file.
            split (str): Data split - "train", "val", or None for all data.
            n_moves (int, optional): Number of moves to return. Defaults to None.
        """
        if n_moves is not None:
            assert n_moves > 0

        # Open H5 file
        self.h5_file = tb.open_file(data_folder+h5_file, mode="r")
        self.encoded_table = self.h5_file.root.encoded_data
        
        self.length = self.encoded_table.nrows

        self.n_moves = n_moves

    def _parse_data(self, idx):
        # Retrieve data from H5 file for index idx
        entry = self.encoded_table[idx]
        
        turns = tf.convert_to_tensor([entry["turn"]], dtype=tf.int32)
        white_kingside_castling_rights = tf.convert_to_tensor([entry["white_kingside_castling_rights"]], dtype=tf.int32)
        white_queenside_castling_rights = tf.convert_to_tensor([entry["white_queenside_castling_rights"]], dtype=tf.int32)
        black_kingside_castling_rights = tf.convert_to_tensor([entry["black_kingside_castling_rights"]], dtype=tf.int32)
        black_queenside_castling_rights = tf.convert_to_tensor([entry["black_queenside_castling_rights"]], dtype=tf.int32)
        board_position = tf.convert_to_tensor(entry["board_position"], dtype=tf.int32)
        moves = tf.convert_to_tensor(entry["moves"][: self.n_moves + 1], dtype=tf.int64)
        length = tf.convert_to_tensor([entry["length"]], dtype=tf.int64)
        
        # Clamp length to be less than or equal to n_moves
        length = tf.minimum(length, self.n_moves)

        return {
            "turns": turns,
            "white_kingside_castling_rights": white_kingside_castling_rights,
            "white_queenside_castling_rights": white_queenside_castling_rights,
            "black_kingside_castling_rights": black_kingside_castling_rights,
            "black_queenside_castling_rights": black_queenside_castling_rights,
            "board_positions": board_position,
            "moves": moves,
            "lengths": length,
        }

    def _generator(self):
        for i in range(self.length):
            yield self._parse_data(i)

    def as_tensorflow_dataset(self):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature={
                "turns": tf.TensorSpec(shape=(1,), dtype=tf.int32),
                "white_kingside_castling_rights": tf.TensorSpec(shape=(1,), dtype=tf.int32),
                "white_queenside_castling_rights": tf.TensorSpec(shape=(1,), dtype=tf.int32),
                "black_kingside_castling_rights": tf.TensorSpec(shape=(1,), dtype=tf.int32),
                "black_queenside_castling_rights": tf.TensorSpec(shape=(1,), dtype=tf.int32),
                "board_positions": tf.TensorSpec(shape=(64,), dtype=tf.int32),
                "moves": tf.TensorSpec(shape=(self.n_moves+1,), dtype=tf.int32),
                "lengths": tf.TensorSpec(shape=(1,), dtype=tf.int32),
            }
        )

"""import yaml
with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

batch_size = cfg['dataloading']['batch_size']

# Example of how to use the TensorFlow Dataset:
dataset = ChessDataset(
    data_folder="path/to/data_folder",
    h5_file="data.h5",
    split="train",
    n_moves=10
).as_tensorflow_dataset()

dataset = dataset.shuffle(buffer_size=int(float(cfg['dataloading']['buffer_size']))).prefetch(tf.data.AUTOTUNE).batch(batch_size).cache()"""


"""# Iterate over the dataset
for batch in dataset:
    print(batch)  # Each batch will be a dictionary with tensors"""
