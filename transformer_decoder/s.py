import chess.pgn
from datetime import datetime
import io
import chess

# Sample PGN
pgn = """
[Event "Rated Bullet tournament https://lichess.org/tournament/yc1WW2Ox"]
[Site "https://lichess.org/PpwPOZMq"]
[Date "2017.04.01"]
[Round "-"]
[White "Abbot"]
[Black "Costello"]
[Result "0-1"]
[UTCDate "2017.04.01"]
[UTCTime "11:32:01"]
[WhiteElo "2100"]
[BlackElo "2000"]
[WhiteRatingDiff "-4"]
[BlackRatingDiff "+1"]
[WhiteTitle "FM"]
[ECO "B30"]
[Opening "Sicilian Defense: Old Sicilian"]
[TimeControl "300+0"]
[Termination "Time forfeit"]

1. e4 { [%eval 0.17] [%clk 0:00:30] } 1... c5 { [%eval 0.19] [%clk 0:00:30] }
2. Nf3 { [%eval 0.25] [%clk 0:00:29] } 2... Nc6 { [%eval 0.33] [%clk 0:00:30] }
3. Bc4 { [%eval -0.13] [%clk 0:00:28] } 3... e6 { [%eval -0.04] [%clk 0:00:30] }
4. c3 { [%eval -0.4] [%clk 0:00:27] } 4... b5? { [%eval 1.18] [%clk 0:00:30] }
5. Bb3?! { [%eval 0.21] [%clk 0:00:26] } 5... c4 { [%eval 0.32] [%clk 0:00:29] }
6. Bc2 { [%eval 0.2] [%clk 0:00:25] } 6... a5 { [%eval 0.6] [%clk 0:00:29] }
7. d4 { [%eval 0.29] [%clk 0:00:23] } 7... cxd3 { [%eval 0.6] [%clk 0:00:27] }
8. Qxd3 { [%eval 0.12] [%clk 0:00:22] } 8... Nf6 { [%eval 0.52] [%clk 0:00:26] }
9. e5 { [%eval 0.39] [%clk 0:00:21] } 9... Nd5 { [%eval 0.45] [%clk 0:00:25] }
10. Bg5?! { [%eval -0.44] [%clk 0:00:18] } 10... Qc7 { [%eval -0.12] [%clk 0:00:23] }
11. Nbd2?? { [%eval -3.15] [%clk 0:00:14] } 11... h6 { [%eval -2.99] [%clk 0:00:23] }
12. Bh4 { [%eval -3.0] [%clk 0:00:11] } 12... Ba6? { [%eval -0.12] [%clk 0:00:23] }
13. b3?? { [%eval -4.14] [%clk 0:00:02] } 13... Nf4? { [%eval -2.73] [%clk 0:00:21] } 0-1
"""

import requests
import time
import zstandard as zstd
from urllib3.exceptions import ProtocolError
import sys
from utils import is_berserk_game, get_move_numbers

m=0
last_processed_game_index = 0
url='https://database.lichess.org/standard/lichess_db_standard_rated_2024-08.pgn.zst'

while True:
    try:
        start_time = time.time()
        with requests.Session() as session:
            response = session.get(url, stream=True)
            response.raise_for_status()

            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(response.raw) as stream_reader:
                decompressed_file = io.TextIOWrapper(stream_reader, encoding='utf-8', errors='replace')

                # Skip to the last processed game
                start = time.time()
                for _ in range(last_processed_game_index):
                    m = chess.pgn.read_game(decompressed_file)
                elapsed = time.time() - start
                if last_processed_game_index > 0:
                    print(f"time taken to skip {last_processed_game_index} games: {elapsed:.4f}s")

                game = chess.pgn.read_game(decompressed_file)
                start_data_time = time.time()
                while game:
                    
                    white_rating = int(game.headers.get("WhiteElo"))
                    black_rating = int(game.headers.get("BlackElo"))
                    
                    # Initialize separate lists for White and Black clock times
                    white_clock_times = []
                    black_clock_times = []

                    # Iterate through all the moves in the game
                    for node in game.mainline():
                        # Check if the node contains a clock time (if the move has a comment with %clk)
                        clock_time = node.clock()
                        
                        # If clock time is found, append to the correct list based on the player's turn
                        if clock_time is not None:
                            if node.turn() == False:
                                #False means white's turn
                                white_clock_times.append(clock_time)
                            else:
                                black_clock_times.append(clock_time)
                                
                    # Initialize an empty list to store the move counts
                    move_counts = []

                    # Track the current move number for White and Black
                    white_move_count = 0
                    black_move_count = 0

                    # Iterate through the moves in the game
                    for i, move in enumerate(game.mainline_moves()):
                        if i % 2 == 0:  # White's move (even-indexed)
                            white_move_count += 1
                            move_counts.append(white_move_count - 1)  # Append the move number for White
                        else:  # Black's move (odd-indexed)
                            black_move_count += 1
                            move_counts.append(black_move_count - 1)  # Append the move number for Black


                    # Initialize separate lists for White and Black elapsed times
                    white_elapsed_times = []
                    black_elapsed_times = []

                    # Variables to store previous clock times
                    previous_white_time = None
                    previous_black_time = None

                    # Iterate through all the moves in the game
                    for node in game.mainline():
                        # Check if the node contains a clock time (if the move has a comment with %clk)
                        clock_time = node.clock()
                        
                        # If a clock time is found, calculate elapsed time
                        if clock_time is not None:
                            #False means white's turn
                            if node.turn() == False:
                                # If this is White's move, calculate the elapsed time for White
                                if previous_white_time is not None:
                                    elapsed_time = previous_white_time - clock_time
                                    white_elapsed_times.append(elapsed_time)
                                previous_white_time = clock_time  # Update the previous clock time for White
                            else:
                                # If this is Black's move, calculate the elapsed time for Black
                                if previous_black_time is not None:
                                    elapsed_time = previous_black_time - clock_time
                                    black_elapsed_times.append(elapsed_time)
                                previous_black_time = clock_time  # Update the previous clock time for Black

                    termination = game.headers.get("Termination")
                    time_control = game.headers.get("TimeControl")
                    
                    if termination != 'Abandoned' and  time_control!='-' and len(white_clock_times)>=5:
                        base_time, increment_time = map(int, time_control.split("+"))
                        if base_time>0.5*int(white_clock_times[0]):
                            base_time, increment_time = map(int, time_control.split("+"))
                            white_elapsed_times.insert(0, 0)
                            black_elapsed_times.insert(0, 0)
                    
                    if white_rating==1527 and black_rating==1500:
                        # Print the lists of clock times for White and Black
                        print("White clock times:", len(white_clock_times))
                        print("Black clock times:", len(black_clock_times))
                        
                        # Print the lists of elapsed times for White and Black
                        print("White elapsed times:", len(white_elapsed_times))
                        print("Black elapsed times:", len(black_elapsed_times))
                        
                        # Print the lists of clock times for White and Black
                        print("White clock times:", white_clock_times)
                        print("Black clock times:", black_clock_times)
                        
                        # Print the lists of elapsed times for White and Black
                        print("White elapsed times:", white_elapsed_times)
                        print("Black elapsed times:", black_elapsed_times)
                        
                        print(game)
                        print('\n\n')
                    
                    
                    
                    """m+=1
                    if m==2:
                        sys.exit()"""
                    
                    game = chess.pgn.read_game(decompressed_file)
    except ProtocolError:
            import random
            delay = random.randint(1, 10)
            print(f"ProtocolError encountered. Retrying in {delay}s...")
            time.sleep(delay)


