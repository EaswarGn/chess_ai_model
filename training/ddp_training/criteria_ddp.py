import torch
import torch.nn.functional as F
import torch.nn as nn


class LabelSmoothedCE(torch.nn.Module):
    """
    Cross Entropy loss with label-smoothing as a form of regularization.

    See "Rethinking the Inception Architecture for Computer Vision",
    https://arxiv.org/abs/1512.00567
    """

    def __init__(self, DEVICE, eps, n_predictions):
        """
        Init.

        Args:

            eps (float): Smoothing co-efficient. 

            n_predictions (int): Number of predictions expected per
            datapoint, or length of the predicted sequence.
        """
        super(LabelSmoothedCE, self).__init__()
        self.eps = eps
        self.DEVICE = DEVICE
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
            .to(self.DEVICE)
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
    def __init__(self, CONFIG, device):
        super().__init__()
        self.device = device
        
        self.loss_weights = CONFIG.LOSS_WEIGHTS
        
        self.loss_functions = {
            'move_loss': CONFIG.move_loss(DEVICE=device, eps=CONFIG.LABEL_SMOOTHING, n_predictions=CONFIG.N_MOVES),
            'time_loss': CONFIG.move_time_loss,
            'moves_until_end_loss': CONFIG.moves_until_end_loss,
            'categorical_game_result_loss': CONFIG.categorical_game_result_loss
        }

    def forward(self, predictions, targets):
        individual_losses = {}

        for i, key in enumerate(self.loss_functions):
            loss_fn = self.loss_functions[key]
            
            if loss_fn is None:
                loss = torch.tensor(0.0)
                individual_losses[key] = loss
                continue
            
            if key == 'move_loss':
                loss = loss_fn(predictions['from_squares'].to(torch.int64), targets["from_squares"].to(torch.int64), targets["lengths"]) + \
                       loss_fn(predictions['to_squares'].to(torch.int64), targets["to_squares"].to(torch.int64), targets["lengths"])
            if key == 'time_loss':  # Fix indentation here
                loss = loss_fn(
                    predictions['move_time'].float(), 
                    targets['move_time'].float()
                )
                
            if key == 'result_loss':
                loss = loss_fn(
                    predictions['game_result'].float(), 
                    targets['game_result'].float()
                )
            if key == 'moves_until_end_loss':
                loss = loss_fn(
                    predictions['moves_until_end'].float(), 
                    targets['moves_until_end'].float()
                )
            if key == 'categorical_game_result_loss':
                targets['categorical_result'] = targets['categorical_result'].squeeze(1)
                loss = loss_fn(
                    predictions['categorical_game_result'].float(), 
                    targets['categorical_result']
                )
                
            individual_losses[key] = loss
            
        individual_losses['result_loss'] = torch.tensor(0.0)

        total_loss = 0.0
        for i, key in enumerate(individual_losses):
            total_loss += self.loss_weights[key+'_weight'] * individual_losses[key]

        return total_loss, individual_losses
