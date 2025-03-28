import requests
import json
import argparse
from tqdm import tqdm
import chess.pgn
import sys
import io

HEADERS = {"User-Agent": "MyChessBot/1.0"}

def get_total_games(archive_urls):
    pbar = tqdm(desc="Counting games", total=len(archive_urls))
    total_games = 0
    for archive_url in archive_urls:
        games_response = requests.get(archive_url, headers=HEADERS)
        games = games_response.json().get("games", [])
        total_games += len(games)
        pbar.update(1)
    pbar.close()
    return total_games

def download_games(username, train_output_file, validation_output_file, train_ratio):
    
    archive_urls = requests.get(f"https://api.chess.com/pub/player/{username}/games/archives", headers=HEADERS).json()["archives"]
    
    pbar = None
    total = get_total_games(archive_urls)
    pbar = tqdm(desc="Downloading games", total=total)
        
    total_train_games = int(train_ratio * total)
    
    with open(train_output_file, "a") as file:
        for archive_url in archive_urls:
            games_response = requests.get(archive_url, headers=HEADERS)
            games = games_response.json().get("games", [])
            
            # Extract and save PGN data
            for game in games:
                if "pgn" in game:
                    curr_game = chess.pgn.read_game(io.StringIO(game["pgn"]))
                    
                    if (curr_game.headers.get("Event") != 'Live Chess' or
                        curr_game.headers.get("Variant") is not None or 
                        curr_game.headers.get("WhiteElo") == '?' or 
                        curr_game.headers.get("BlackElo") == '?' or
                        curr_game.headers.get("TimeControl") == '-' or
                        curr_game.headers.get("FEN") is not None or 
                        'abandon' in curr_game.headers.get("Termination")):
                        continue
                    else:
                        pass
                            
                    
                    if pbar.n <= total_train_games:
                        file.write(game["pgn"] + "\n")
                        pbar.update(1)
                    
                    if pbar.n >= total_train_games:
                        with open(validation_output_file, "a") as validation_file:
                            validation_file.write(game["pgn"] + "\n")
                            pbar.update(1)
        pbar.close()

                        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Chess.com games for a user.")
    parser.add_argument("--username", required=True, help="Chess.com username (e.g., --username='Hikaru')")
    parser.add_argument("--train_output", required=True, help="Output PGN file for training data (e.g., --train_output='games_train.pgn')")
    parser.add_argument("--validation_output", required=True, help="Output PGN file for validation games (e.g., --validation_output='games_validation.pgn')")
    parser.add_argument("--train_ratio", type=float, default=0, help="Percent of games to be used as training data, rest will be used for validation/testing")
    args = parser.parse_args()

    download_games(args.username, args.train_output, args.validation_output, args.train_ratio)