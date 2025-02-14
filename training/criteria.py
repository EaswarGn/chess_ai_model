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
                 move_weight=1.0, 
                 time_weight=0.5, 
                 result_weight=1.0,
                 moves_until_end_weight=0.5,
                 temperature=1.0,
                 criterion=None,
                 log_space=True):
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
        self.move_loss = criterion  # for from/to squares
        self.time_loss = nn.L1Loss()  # for move time prediction
        self.result_loss = nn.L1Loss()  # for game result prediction
        self.moves_until_end_loss = nn.L1Loss()
        
        # Option to use log-space parametrization for more stable learning
        self.log_space = log_space
        
        if log_space:
            # Initialize log weights 
            self.log_weights = nn.Parameter(torch.log(torch.tensor([
                move_weight,   # from/to squares
                time_weight,   # move time
                result_weight,  # game result
                moves_until_end_weight
            ])))
        else:
            # Direct weight parametrization 
            self.weights = nn.Parameter(torch.tensor([
                move_weight,   # from/to squares
                time_weight,   # move time
                result_weight,  # game result
                moves_until_end_weight
            ]))
        
        self.temperature = temperature

    def forward(self, predictions, targets):
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
        )
        
        """# Stack losses
        losses = torch.stack([
            move_loss,
            time_loss,
            result_loss,
            moves_until_end_loss
        ])
        
        # Dynamic weight computation
        if self.log_space:
            # Exponentiate log weights for positive values
            weights = F.softmax(self.log_weights / self.temperature, dim=0)
        else:
            # Use softmax directly on weights
            weights = F.softmax(self.weights / self.temperature, dim=0)
        
        # Compute weighted total loss
        total_loss = torch.sum(losses * weights)"""
        
        total_loss = move_loss + time_loss + result_loss + moves_until_end_loss
        
        
        return total_loss, {
            'move_loss': move_loss,
            'time_loss': time_loss,
            'result_loss': result_loss,
            'moves_until_end_loss': moves_until_end_loss,
            #'weights': weights.detach()
        }