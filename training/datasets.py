import os
import json
import torch
import argparse
import tables as tb
from torch.utils.data import Dataset

from configs import import_config
from time_controls import time_controls_encoded


class ChessDataset(Dataset):
    def __init__(self, data_folder, h5_file, split, n_moves=None, **unused):
        """
        Init.

        Args:

            datasets (list): A list of tuples, each containing:

                (dict): A data configuration representing the
                dataset.

                (str): The data split. One of "train", "val", or
                None, which means that all datapoints will be
                included.

            n_moves (int, optional): Number of moves into the future to
            return. Defaults to None, which means that all moves in the
            H5 data column will be returned.
        """
        if n_moves is not None:
            assert n_moves > 0

        # Open table in H5 file
        #self.h5_file = tb.open_file(os.path.join(data_folder, h5_file), mode="r")
        #self.encoded_table = self.h5_file.root.encoded_data
        self.h5_file = os.path.join(data_folder, h5_file)
        self.encoded_table = self.h5_file.root.encoded_data
        
        self.n_moves = n_moves
        self.m = 0

    def __getitem__(self, i):
        turns = torch.IntTensor([self.encoded_table[i]["turn"]])
        white_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[i]["white_kingside_castling_rights"]]
        )  # (1)
        white_queenside_castling_rights = torch.IntTensor(
            [self.encoded_table[i]["white_queenside_castling_rights"]]
        )  # (1)
        black_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[i]["black_kingside_castling_rights"]]
        )  # (1)
        black_queenside_castling_rights = torch.IntTensor(
            [self.encoded_table[i]["black_queenside_castling_rights"]]
        )  # (1)
        board_position = torch.IntTensor(
            self.encoded_table[i]["board_position"]
        )  # (64)
        moves = torch.LongTensor(
            self.encoded_table[i]["moves"][: self.n_moves + 1]
        )  # (n_moves + 1)
        length = torch.LongTensor(
            [self.encoded_table[i]["length"]]
        ).clamp(
            max=self.n_moves
        )  # (1), value <= n_moves
        

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

    def __len__(self):
        return self.encoded_table.nrows


class ChessDatasetFT(Dataset):
    def __init__(self, data_folder, h5_file, split, **unused):
        """
        Init.

        Args:

            data_folder (str): The folder containing the H5 and splits
            files.

            h5_file (str): The H5 file.

            split (str): The data split. One of "train", "val", None.
            Defaults to None, which means that all datapoints will be
            included.
        """
        # Open table in H5 file
        self.h5_file = tb.open_file(os.path.join(data_folder, h5_file), mode="r")
        self.encoded_table = self.h5_file.root.encoded_data
        self.human_table = self.h5_file.root.data
        self.len = self.encoded_table.nrows
        #self.split = split
        
        self.first_index = 0

    def __getitem__(self, i):
        turns = torch.IntTensor([self.encoded_table[self.first_index + i]["turn"]])
        white_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["white_kingside_castling_rights"]]
        )  # (1)
        white_queenside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["white_queenside_castling_rights"]]
        )  # (1)
        black_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["black_kingside_castling_rights"]]
        )  # (1)
        black_queenside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["black_queenside_castling_rights"]]
        )  # (1)
        board_position = torch.IntTensor(
            self.encoded_table[self.first_index + i]["board_position"]
        )  # (64)
        from_square = torch.LongTensor(
            [self.encoded_table[self.first_index + i]["from_square"]]
        )  # (1)
        to_square = torch.LongTensor(
            [self.encoded_table[self.first_index + i]["to_square"]]
        )  # (1)
        length = torch.LongTensor([1])
        
       #new features
        phase = torch.IntTensor(
            [self.encoded_table[i]['phase']]
        )
        result = torch.IntTensor(
            [self.encoded_table[i]['result']]
        )
        
        try:
            base_time = self.encoded_table[i]['base_time']
            increment_time = self.encoded_table[i]['increment_time']
            time_control = f'{base_time}+{increment_time}'
            
            # Attempt to look up the time control encoding
            time_control = torch.LongTensor([time_controls_encoded[time_control]])

        except KeyError as e:
            # Handle the KeyError
            print(f"KeyError: Missing key {e} in encoded_table or time_controls_encoded")
            # You can either provide a default value or skip the iteration, etc.
            base_time = None
            increment_time = None
            time_control = torch.LongTensor([0])  # Or whatever default you need

        
        """white_remaining_time = torch.FloatTensor(
            [self.encoded_table[i]['white_remaining_time']]
        )
        black_remaining_time = torch.FloatTensor(
            [self.encoded_table[i]['black_remaining_time']]
        )"""
        white_remaining_time = torch.IntTensor(
            [self.human_table[i]['white_remaining_time']]
        )
        black_remaining_time = torch.IntTensor(
            [self.human_table[i]['black_remaining_time']]
        )
        """white_rating = torch.IntTensor(
            [self.encoded_table[i]['white_rating']-1]
        )
        black_rating = torch.IntTensor(
            [self.encoded_table[i]['black_rating']-1]
        )"""
        white_rating = torch.IntTensor(
            [self.human_table[i]['white_rating']-1]
        )
        black_rating = torch.IntTensor(
            [self.human_table[i]['black_rating']-1]
        )
        """time_spent_on_move = torch.FloatTensor(
            [self.encoded_table[i]['time_spent_on_move']]
        )"""
        time_spent_on_move = torch.IntTensor(
            [self.human_table[i]['time_spent_on_move']]
        )
        move_number = torch.IntTensor(
            [self.encoded_table[i]['move_number']]
        )
        num_legal_moves = torch.IntTensor(
            [self.encoded_table[i]['num_legal_moves']]
        )
        
        white_material_value = torch.IntTensor(
            [self.encoded_table[i]['white_material_value']]
        )
        
        black_material_value = torch.IntTensor(
            [self.encoded_table[i]['black_material_value']]
        )
        
        material_difference = torch.IntTensor(
            [self.encoded_table[i]['material_difference']]
        )
        
        moves_until_end = torch.IntTensor(
            [self.encoded_table[i]['moves_until_end']]
        )
        
        return {
            "turns": turns,
            "white_kingside_castling_rights": white_kingside_castling_rights,
            "white_queenside_castling_rights": white_queenside_castling_rights,
            "black_kingside_castling_rights": black_kingside_castling_rights,
            "black_queenside_castling_rights": black_queenside_castling_rights,
            "board_positions": board_position,
            "from_squares": from_square,
            "to_squares": to_square,
            "lengths": length,
            "phase": phase,
            "result": result,
            "time_control": time_control,
            "white_remaining_time": white_remaining_time,
            "black_remaining_time": black_remaining_time,
            "white_rating": white_rating,
            "black_rating": black_rating,
            "move_time": time_spent_on_move,
            "move_number": move_number,
            "num_legal_moves": num_legal_moves,
            "white_material_value": white_material_value,
            "black_material_value": black_material_value,
            "material_difference": material_difference,
            "moves_until_end": moves_until_end
        }

    def __len__(self):
        return self.len



if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name)

    # Dataset
    dataset = CONFIG.DATASET(
        data_folder=CONFIG.DATA_FOLDER,
        h5_file=CONFIG.H5_FILE,
        split="train",
        n_moves=5,
    )
    print(len(dataset))
    print(dataset[17])
    dataset.h5_file.close()