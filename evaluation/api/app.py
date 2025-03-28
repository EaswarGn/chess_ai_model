from flask import Flask, request, jsonify
import joblib
import numpy as np
from utils import load_model, get_model_inputs, get_move, get_move_probabilities, get_all_move_probabilities
from configs import import_config
import chess
import torch.nn.functional as F
import random

CONFIG = import_config('individual_model')
CONFIG = CONFIG.CONFIG()
app = Flask(__name__)

# Load the model
model = load_model(CONFIG)
pondering_time_model = load_model(import_config('pondering_time_model').CONFIG())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        board = chess.Board(request.json['board'])
        time_control = request.json['time_control']
        white_remaining_time = int(request.json['white_remaining_time'])
        black_remaining_time = int(request.json['black_remaining_time'])
        white_rating = int(request.json['white_rating'])
        black_rating = int(request.json['black_rating'])

        inputs = get_model_inputs(board,
                time_control=time_control,
                white_remaining_time=white_remaining_time,
                black_remaining_time=black_remaining_time,
                white_rating=white_rating,
                black_rating=black_rating
        )
        
        
        predictions = model(inputs)
        pondering_time_pred = pondering_time_model(inputs)
        
        
        model_move = get_move(board, predictions)
        predictions['move_time'] = abs(round(pondering_time_pred['move_time'][0].item(), 4))
        all_move_probabilites = get_all_move_probabilities(board, predictions)
        legal_move_probabilities = get_move_probabilities(board, predictions)
        
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
