import tables as tb
import time
from collections import defaultdict
import sys
import torch

h5_file = tb.open_file('data.h5', mode="r")
encoded_table = h5_file.root.encoded_data

print("total rows:",encoded_table.nrows)

"""unique_time_controls = defaultdict(int)
for i in range(1):
    h5_file = tb.open_file(f'data/epoch_{i+1}/train_data/data.h5', mode="r")
    encoded_table = h5_file.root.encoded_data
    for i in range(encoded_table.nrows):
        base_time = encoded_table[i]['base_time']
        increment_time = encoded_table[i]['increment_time']
        time_control = f'{base_time}+{increment_time}'
        unique_time_controls[time_control] += 1
print(dict(unique_time_controls))
sys.exit()"""


"""start = time.time()
opening_count = 0
middlegame_count = 0
endgame_count = 0
for i in range(encoded_table.nrows):
    if encoded_table[i]['phase'] == 2:
        opening_count+=1
    if encoded_table[i]['phase'] == 3:
        middlegame_count+=1
    if encoded_table[i]['phase'] == 4:
        endgame_count+=1
print("total opening positions: ",opening_count)
print("total middlegame positions: ",middlegame_count)
print("total endgame positions: ",endgame_count)
print(f"table looped through in {(time.time()-start):.4f}s\n")"""

table = encoded_table


row_data = table[35]
print(f"board_position: {row_data['board_position']}")
"""if is_human_table:
    print(f"raw_fen: {row_data['raw_fen']}")"""
print(f"turn: {row_data['turn']}")
print(f"white_kingside_castling_rights: {row_data['white_kingside_castling_rights']}")
print(f"white_queenside_castling_rights: {row_data['white_queenside_castling_rights']}")
print(f"black_kingside_castling_rights: {row_data['black_kingside_castling_rights']}")
print(f"black_queenside_castling_rights: {row_data['black_queenside_castling_rights']}")
print(f"moves: {row_data['moves']}")
print(f"length: {row_data['length']}")
print(f"from_square: {row_data['from_square']}")
print(f"to_square: {row_data['to_square']}")
print(f"phase: {row_data['phase']}")
print(f"result: {row_data['result']}")
print(f"base_time: {row_data['base_time']}")
print(f"increment_time: {row_data['increment_time']}")
#print(f"remaining_time: {row_data['remaining_time']}")
print(f"white_remaining_time: {row_data['white_remaining_time']}")
print(f"black_remaining_time: {row_data['black_remaining_time']}")
print(f"white_rating: {row_data['white_rating']}")
print(f"black_rating: {row_data['black_rating']}")
print(f"time_spent_on_move: {row_data['time_spent_on_move']}")
print(f"move_number: {row_data['move_number']}")
print(f"num_legal_moves: {row_data['num_legal_moves']}")
print(f"white_material_value: {row_data['white_material_value']}")
print(f"black_material_value: {row_data['black_material_value']}")
print(f"material_difference: {row_data['material_difference']}")
print(f"moves_until_end: {row_data['moves_until_end']}")


start = time.time()
turns = torch.IntTensor([encoded_table[0]["turn"]])
white_kingside_castling_rights = torch.IntTensor(
    [encoded_table[0]["white_kingside_castling_rights"]]
)  # (1)
white_queenside_castling_rights = torch.IntTensor(
    [encoded_table[0]["white_queenside_castling_rights"]]
)  # (1)
black_kingside_castling_rights = torch.IntTensor(
    [encoded_table[0]["black_kingside_castling_rights"]]
)  # (1)
black_queenside_castling_rights = torch.IntTensor(
    [encoded_table[0]["black_queenside_castling_rights"]]
)  # (1)
board_position = torch.IntTensor(
   encoded_table[0]["board_position"]
)  # (64)
from_square = torch.LongTensor(
    [encoded_table[0]["from_square"]]
)  # (1)
to_square = torch.LongTensor(
    [encoded_table[0]["to_square"]]
)  # (1)
length = torch.LongTensor([1])

#new features
phase = torch.IntTensor(
    [encoded_table[0]['phase']-2]
)
result = torch.IntTensor(
    [encoded_table[0]['result']]
)

categorical_result = torch.LongTensor(
    [encoded_table[0]['result']]
)

try:
    base_time = encoded_table[0]['base_time']
    increment_time = encoded_table[0]['increment_time']
    time_control = f'{base_time}+{increment_time}'
    
    # Attempt to look up the time control encoding
    time_control = torch.LongTensor([0])

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
white_remaining_time = torch.FloatTensor(
    [encoded_table[0]['white_remaining_time']]
)
black_remaining_time = torch.FloatTensor(
    [encoded_table[0]['black_remaining_time']]
)
"""white_rating = torch.IntTensor(
    [self.encoded_table[i]['white_rating']-1]
)
black_rating = torch.IntTensor(
    [self.encoded_table[i]['black_rating']-1]
)"""
white_rating = torch.IntTensor(
    [encoded_table[0]['white_rating']-1]
)
black_rating = torch.IntTensor(
    [encoded_table[0]['black_rating']-1]
)
"""time_spent_on_move = torch.FloatTensor(
    [self.encoded_table[i]['time_spent_on_move']]
)"""
time_spent_on_move = torch.FloatTensor(
    [encoded_table[0]['time_spent_on_move']/100]
)
move_number = torch.IntTensor(
    [encoded_table[0]['move_number']]
)
num_legal_moves = torch.IntTensor(
    [encoded_table[0]['num_legal_moves']]
)

white_material_value = torch.IntTensor(
    [encoded_table[0]['white_material_value']]
)

black_material_value = torch.IntTensor(
    [encoded_table[0]['black_material_value']]
)

material_difference = torch.IntTensor(
    [encoded_table[0]['material_difference']]
)

moves_until_end = torch.FloatTensor(
    [encoded_table[0]['moves_until_end']/100]
)
print(f"time taken to create tensors: {time.time() - start}s")


h5_file.close()
