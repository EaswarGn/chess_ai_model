import torch
import torch.nn.functional as F
import torch.nn as nn

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class LabelSmoothedCE(torch.nn.Module):
    """
    Cross Entropy loss with label-smoothing as a form of regularization.

    See "Rethinking the Inception Architecture for Computer Vision",
    https://arxiv.org/abs/1512.00567
    """

    def __init__(self, eps, n_predictions):
        """
        Init.

        Args:

            eps (float): Smoothing co-efficient. 

            n_predictions (int): Number of predictions expected per
            datapoint, or length of the predicted sequence.
        """
        super(LabelSmoothedCE, self).__init__()
        self.eps = eps
        self.indices = torch.arange(n_predictions).unsqueeze(0).to(DEVICE)  # (1, n_predictions)
        self.indices.requires_grad = False

    def forward(self, predicted, targets, lengths):
        """
        Forward prop.

        Args:

            predicted (torch.FloatTensor): The predicted probabilities,
            of size (N, n_predictions, vocab_size).

            targets (torch.LongTensor): The actual targets, of size (N,
            n_predictions).

            lengths (torch.LongTensor): The true lengths of the
            prediction sequences, not including special tokens, of size
            (N, 1).

        Returns:

            torch.Tensor: The mean label-smoothed cross-entropy loss, a
            scalar.
        """
        # Remove pad-positions and flatten
        predicted = predicted[
            self.indices < lengths
        ]  # (sum(lengths), vocab_size)
        targets = targets[self.indices < lengths]  # (sum(lengths))

        # "Smoothed" one-hot vectors for the gold sequences
        target_vector = (
            torch.zeros_like(predicted)
            .scatter(dim=1, index=targets.unsqueeze(1), value=1.0)
            .to(DEVICE)
        )  # (sum(lengths), vocab_size), one-hot
        target_vector = target_vector * (
            1.0 - self.eps
        ) + self.eps / target_vector.size(
            1
        )  # (sum(lengths), vocab_size), "smoothed" one-hot

        # Compute smoothed cross-entropy loss
        loss = (-1 * target_vector * F.log_softmax(predicted, dim=1)).sum(
            dim=1
        )  # (sum(lengths))

        # Compute mean loss
        loss = torch.mean(loss)

        return loss
    


class MultiTaskChessLoss(nn.Module):
    def __init__(self, 
                 #loss_weights,
                 CONFIG,
                 #criterion=None,
        ):
        """
        Multi-task loss with learnable, dynamically balanced weights.
        
        Args:
            move_weight (float): Initial move loss weight
            time_weight (float): Initial time loss weight
            result_weight (float): Initial result loss weight
            temperature (float): Softness of weight scaling
            criterion (callable): Custom loss function for move prediction
            log_space (bool): Whether to use log-space weight parametrization
        """
        super().__init__()
        """self.move_loss = criterion  # for from/to squares
        self.time_loss = nn.L1Loss()  # for move time prediction
        self.result_loss = nn.L1Loss()  # for game result prediction
        self.moves_until_end_loss = nn.L1Loss()"""
        self.loss_weights = CONFIG.LOSS_WEIGHTS
        self.loss_functions = CONFIG.LOSSES
        

    def forward(self, predictions, targets):
        
        individual_losses = {
            'move_loss': torch.tensor(0.0),
            'move_time_loss': torch.tensor(0.0),
            'game_result_loss': torch.tensor(0.0),
            'moves_until_end_loss': torch.tensor(0.0)
        }
        
        for key in self.loss_functions:
            if key == 'move_loss':
                individual_losses[key] = self.loss_functions[key](
                    predicted=predictions['from_squares'],
                    targets=targets["from_squares"],
                    lengths=targets["lengths"],
                ) + self.loss_functions[key](
                    predicted=predictions['to_squares'],
                    targets=targets["to_squares"],
                    lengths=targets["lengths"],
                )

            if key == 'move_time_loss':  # Fix indentation here
                individual_losses[key] = self.loss_functions[key](
                    predictions['move_time'].float(), 
                    targets['move_time'].float()
                )
                
            if key == 'game_result_loss':
                individual_losses[key] = self.loss_functions[key](
                    predictions['game_result'].float(), 
                    targets['game_result'].float()
                )
            if key == 'moves_until_end_loss':
                individual_losses[key] = self.loss_functions[key](
                    predictions['moves_until_end'].float(), 
                    targets['moves_until_end'].float()
                )

        loss_details = {
            'result_loss': individual_losses['game_result_loss'],
            'time_loss': individual_losses['move_time_loss'],
            'move_loss': individual_losses['move_loss'],
            'moves_until_end_loss': individual_losses['moves_until_end_loss']
        }

        loss_weights = self.loss_weights
        total_loss = 0.0
        for key in individual_losses:
            total_loss += individual_losses[key]*loss_weights[f'{key}_weight']
        
        return total_loss, loss_details
                
        """
        # Compute individual losses
        move_loss = self.move_loss(
            predicted=predictions['from_squares'],
            targets=targets["from_squares"],
            lengths=targets["lengths"],
        ) + self.move_loss(
            predicted=predictions['to_squares'],
            targets=targets["to_squares"],
            lengths=targets["lengths"],
        )
        
        time_loss = self.time_loss(
            predictions['move_time'].float(), 
            targets['move_time'].float()
        )
        
        result_loss = self.result_loss(
            predictions['game_result'].float(), 
            targets['game_result'].float()
        )
        
        moves_until_end_loss = self.moves_until_end_loss(
            predictions['moves_until_end'].float(), 
            targets['moves_until_end'].float()
        )"""
        
        
        
        """return total_loss, {
            'move_loss': move_loss,
            'time_loss': time_loss,
            'result_loss': result_loss,
            'moves_until_end_loss': moves_until_end_loss,
        }"""