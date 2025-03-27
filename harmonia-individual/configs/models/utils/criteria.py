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

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', device='cpu'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.device = device

        # Set default alpha if not provided
        if alpha is None:
            self.alpha = torch.tensor([0.25, 1.25, 0.25], device=self.device)
        else:
            self.alpha = alpha.to(self.device) if isinstance(alpha, torch.Tensor) else torch.tensor(alpha, device=self.device)

    def forward(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        # Traditional cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get the probability for the correct class
        pt = torch.exp(-ce_loss)

        # Apply focal weighting
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting for class imbalance
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.gather(0, targets)
                focal_weight = alpha_t * focal_weight
            else:
                focal_weight = self.alpha * focal_weight

        # Apply weighting to CE loss
        loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
