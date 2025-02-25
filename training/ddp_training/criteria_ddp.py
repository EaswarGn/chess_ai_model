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
        
        # Initialize learnable weights (log-space for stability)
        self.log_vars = nn.Parameter(torch.zeros(len(CONFIG.LOSS_WEIGHTS)), requires_grad=True)
        
        self.loss_functions = {
            'move_loss': LabelSmoothedCE(DEVICE=device, eps=CONFIG.LABEL_SMOOTHING, n_predictions=CONFIG.N_MOVES),
            'move_time_loss': nn.L1Loss(),
            'moves_until_end_loss': nn.L1Loss(),
            'categorical_game_result_loss': nn.CrossEntropyLoss()
        }

    def forward(self, predictions, targets):
        individual_losses = {}

        for i, key in enumerate(self.loss_functions):
            loss_fn = self.loss_functions[key]
            
            if key == 'move_loss':
                loss = loss_fn(predictions['from_squares'], targets["from_squares"], targets["lengths"]) + \
                       loss_fn(predictions['to_squares'], targets["to_squares"], targets["lengths"])
            else:
                loss = loss_fn(predictions[key].float(), targets[key].float())
                
            individual_losses[key] = loss

        # Compute dynamically weighted loss
        total_loss = 0.0
        for i, key in enumerate(individual_losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * individual_losses[key] + self.log_vars[i]
            total_loss += weighted_loss

        return total_loss, individual_losses
