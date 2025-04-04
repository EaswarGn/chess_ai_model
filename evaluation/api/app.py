from flask import Flask, request, jsonify
import joblib
import numpy as np
from utils import load_model, get_model_inputs, get_move, get_move_probabilities, get_all_move_probabilities, convert_uci_to_san
from configs import import_config
import chess
import torch.nn.functional as F
import random
import torch
import time
import chess.engine
import os

CONFIG = import_config('individual_model')
CONFIG = CONFIG.CONFIG()
app = Flask(__name__)

# Load the model
model = load_model(CONFIG)
pondering_time_model = load_model(import_config('individual_pondering_time_model').CONFIG())
opening_model = load_model(import_config('opening_model').CONFIG())
engine = chess.engine.SimpleEngine.popen_uci('stockfish')
print("stockfish loaded")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        board = chess.Board(request.json['board'])
        time_control = request.json['time_control']
        white_remaining_time = int(request.json['white_remaining_time'])
        black_remaining_time = int(request.json['black_remaining_time'])
        white_rating = int(request.json['white_rating'])
        black_rating = int(request.json['black_rating'])

        inputs, fullmove_count = get_model_inputs(board,
                time_control=time_control,
                white_remaining_time=white_remaining_time,
                black_remaining_time=black_remaining_time,
                white_rating=white_rating,
                black_rating=black_rating
        )
        
        if fullmove_count<=5:
            predictions = opening_model(inputs)
            pondering_time_pred = predictions['move_time']
            predictions['move_time'] = abs(round(pondering_time_pred[0].item(), 4))
        else:
            predictions = model(inputs)
            pondering_time_pred = pondering_time_model(inputs)
            predictions['move_time'] = abs(round(pondering_time_pred['move_time'][0].item(), 4))
        
        
        model_move = get_move(board, predictions)
        predictions['categorical_game_result'] = torch.softmax(predictions['categorical_game_result'], dim=-1)
        all_move_probabilites = get_all_move_probabilities(board, predictions)
        legal_move_probabilities = get_move_probabilities(board, predictions)
        
        #legal_move_probabilities = convert_uci_to_san(legal_move_probabilities, board)
        total_sum = 0
        for key in legal_move_probabilities:
            total_sum += legal_move_probabilities[key]
        for key in legal_move_probabilities:
            legal_move_probabilities[key]/=total_sum
        
            
        
        
        """probabilities = []
        for key in legal_move_probabilities:
            probabilities.append(legal_move_probabilities[key])
        probabilities = sorted(probabilities, reverse=True)
        prob_range = probabilities[0] - probabilities[5]
        
        if prob_range < 0.4:
            # If the range is less than 0.4, perform sampling
            sampled_index = torch.multinomial(torch.tensor(probabilities[:5]), 1).item()  # .item() to get the scalar value
            sampled_move = list(legal_move_probabilities.keys())[sampled_index]
            
            print(f"In this position: {board.fen()}")
            print(f"the original move was: {model_move}")
            print(f"But the move changed due to softmax sampling: {sampled_move}\n\n\n")
        else:
            # If the range is greater than or equal to 0.4, pick the move with the highest probability
            sampled_move = max(legal_move_probabilities, key=legal_move_probabilities.get)
        
        model_move = sampled_move"""

        """print("analysing")
        start = time.time()
        info = engine.analyse(board, chess.engine.Limit(time=2))  # Set the time limit for the evaluation
        evaluation = info["score"]
        print("Evaluation:", evaluation)
        print(f"time taken to analyze: {time.time()-start}")
        print(info)"""

            
        
        if predictions['move_time'] < 0.5:
            predictions['move_time'] = random.uniform(0.1, 0.5)
        
        #print(get_all_move_probabilities(board, predictions))
        return jsonify(
            {
                'predicted_move': model_move,
                'move_time_spend': abs(round(predictions['move_time'], 4)),
                #'model_evaluation': round(predictions['game_result'][0].item(), 4),
                #'moves_until_game_ends': int(predictions['moves_until_end'][0].item()*100),
                'white_wins_prob': round(predictions['categorical_game_result'][0][2].item(), 4),
                'draw_prob': round(predictions['categorical_game_result'][0][1].item(), 4),
                'black_wins_prob': round(predictions['categorical_game_result'][0][0].item(), 4),
                'legal_move_probabilities': legal_move_probabilities,
                'all_move_probabilities': all_move_probabilites,
            })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
