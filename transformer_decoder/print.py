import tables as tb
import time
from collections import defaultdict
import sys

h5_file = tb.open_file('data/epoch_3/train_data/data.h5', mode="r")
encoded_table = h5_file.root.encoded_data
human_table = h5_file.root.data

print("total rows:",encoded_table.nrows)

unique_time_controls = defaultdict(int)
for i in range(1):
    h5_file = tb.open_file(f'data/epoch_{i+1}/train_data/data.h5', mode="r")
    encoded_table = h5_file.root.encoded_data
    for i in range(encoded_table.nrows):
        base_time = encoded_table[i]['base_time']
        increment_time = encoded_table[i]['increment_time']
        time_control = f'{base_time}+{increment_time}'
        unique_time_controls[time_control] += 1
print(dict(unique_time_controls))
sys.exit()


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

is_human_table = False
if h5_file.root.data == table:
    is_human_table = True


row_data = table[42]
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
h5_file.close()
