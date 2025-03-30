import os
import math
import json
import torch
import pathlib
import torch.nn.functional as F


from pathlib import Path
import torch.nn as nn



def get_all_record_files(directory: str):
    directory = Path(directory).expanduser()
    return [str(file) for file in Path(directory).rglob("*") if file.is_file()]
    
    
def get_lr(step, d_model, warmup_steps, total_steps, schedule="cosine", decay=0.06, batch_size=512, min_lr=1e-5):
    """
    Enhanced learning rate schedule for fine-tuning.
    
    New features:
    - Cosine annealing schedule
    - Lower initial learning rate
    - Minimum learning rate parameter
    """
    if schedule == "vaswani":
        lr = (
            2.0
            * math.pow(d_model, -0.5)
            * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))
        )
    elif schedule == "exp_decay":
        if step <= warmup_steps:
            lr = 1e-4 * step / warmup_steps  # Lowered initial learning rate
        else:
            lr = 1e-4 * ((1 - decay) ** ((step - warmup_steps) / 10000))
    elif schedule == "cosine":
        # Cosine annealing with warm restart characteristics
        if step <= warmup_steps:
            lr = 1e-4 * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = min_lr + 0.5 * (1e-4 - min_lr) * (1 + math.cos(math.pi * progress))
    else:
        raise NotImplementedError

    # Batch size scaling remains the same
    lr = lr * (batch_size // 512)
    
    return max(lr, min_lr)


from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, BadRequestError
import shutil
api = HfApi()

def save_checkpoint(rating, step, model, optimizer, config_name, checkpoint_folder, CONFIG, prefix=""):
    """
    Checkpoint saver. Each save overwrites any previous save.

    Args:

        epoch (int): The epoch number (0-indexed).

        model (torch.nn.Module): The transformer model.

        optimizer (torch.optim.adam.Adam): The optimizer.

        config_name (str): The configuration name.

        checkpoint_folder (str): The folder where checkpoints must be
        saved.

        prefix (str, optional): The checkpoint filename prefix. Defaults
        to "".
    """
    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    optimizer_state_dict = {}
    for key, value in optimizer.state_dict().items():
        if isinstance(value, torch.Tensor):
            optimizer_state_dict[key] = value.cpu()
        elif isinstance(value, dict):
            optimizer_state_dict[key] = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in value.items()}
        else:
            optimizer_state_dict[key] = value
    
    rating = str(rating)
    step = str(step)
    state = {
        "step": step,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "config": CONFIG
    }
    checkpoint_folder = f'{config_name}/{checkpoint_folder}'
    pathlib.Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
    filename = rating + "_step_" + step + ".pt"
    torch.save(state, os.path.join(checkpoint_folder, filename))
    
    os.makedirs(f'{config_name}/logs/checkpoint_logs', exist_ok=True)
    os.makedirs(f"{config_name}/logs/checkpoint_logs/{rating}_step_{step}", exist_ok=True)
    shutil.copy(f"{config_name}/logs/main_log/{os.listdir(f'{config_name}/logs/main_log')[0]}", f"{config_name}/logs/checkpoint_logs/{rating}_step_{step}")
    
    if CONFIG.USE_UPLOAD is True:
        import time

        while True:
            try:
                api.upload_folder(
                    folder_path=f"{config_name}",
                    repo_id=f"codingmonster1234/{config_name}",
                    repo_type="dataset",
                    ignore_patterns="**/logs/*.txt",  # Ignore all text logs
                )
                print("Upload successful!")
                break  # Exit loop when upload is successful
            except (RepositoryNotFoundError, BadRequestError) as e:
                print(f"Error occurred: {type(e).__name__} - {e}")

                if isinstance(e, RepositoryNotFoundError):
                    print("Repository not found. Creating repository...")
                    api.create_repo(f"codingmonster1234/{config_name}", repo_type="dataset")
                elif isinstance(e, BadRequestError):
                    print("Bad request error. Retrying upload...")

                time.sleep(5)  # Wait for 5 seconds before retrying

    
    print("Checkpoint saved.\n")


def change_lr(optimizer, new_lr):
    """
    Change learning rate to a specified value.

    Args:

        optimizer (torch.optim.adam.Adam): Optimizer whose learning rate
        must be changed.

        new_lr (float): New learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def topk_accuracy(logits, targets, other_logits=None, other_targets=None, k=[1, 3, 5]):
    """
    Compute "top-k" accuracies for multiple values of "k".

    Optionally, a second set of logits and targets, for a second
    predicted variable, can be provided. In this case, probabilities
    associated with both sets of logits are combined to arrive at the
    best combinations of both predicted variables. A correct prediction
    occurs when the combination of the targets is present in the top "k"
    predicted combinations.

    Args:

        logits (torch.FloatTensor): Predicted logits, of size (N,
        vocab_size).

        targets (torch.LongTensor): Actual targets, of size (N).

        other_logits (torch.FloatTensor, optional): Predicted logits for
        a second predicted variable, if any, of size (N,
        other_vocab_size). Defaults to None.

        other_targets (torch.LongTensor, optional): Actual targets for a
        second predicted variable, if any, of size (N). Defaults to
        None.

        k (list, optional): Values of "k". Defaults to [1, 3, 5].

    Returns:

        list: "Top-k" accuracies.
    """
    with torch.no_grad():
        batch_size = logits.shape[0]
        if other_logits is not None:
            # Get indices corresponding to top-max(k) scores
            probabilities = F.softmax(logits, dim=-1).unsqueeze(2)  # (N, vocab_size, 1)
            other_probabilities = F.softmax(other_logits, dim=-1).unsqueeze(
                1
            )  # (N, 1, other_vocab_size)
            combined_probabilities = torch.bmm(probabilities, other_probabilities).view(
                batch_size, -1
            )  # (N, vocab_size * other_vocab_size)
            _, flattened_indices = combined_probabilities.topk(
                k=max(k), dim=1
            )  # (N, max(k))
            indices = flattened_indices // other_logits.shape[-1]  # (N, max(k))
            other_indices = flattened_indices % other_logits.shape[-1]  # (N, max(k))

            # Expand targets to the same shape
            targets = targets.unsqueeze(1).expand_as(indices)  # (N, max(k))
            other_targets = other_targets.unsqueeze(1).expand_as(
                other_indices
            )  # (N, max(k))

            # Get correct predictions
            correct_predictions = (indices == targets) * (
                other_indices == other_targets
            )  # (N, max(k))

        else:
            # Get indices corresponding to top-max(k) scores
            _, indices = logits.topk(k=max(k), dim=1)  # (N, max(k))

            # Expand targets to the same shape
            targets = targets.unsqueeze(1).expand_as(indices)  # (N, max(k))

            # Get correct predictions
            correct_predictions = indices == targets  # (N, max(k))

        # Calculate top-k accuracies
        topk_accuracies = [
            correct_predictions[:, :k_value].sum().item() / batch_size for k_value in k
        ]

        return topk_accuracies

def softmax_sampling_accuracy(logits, targets, other_logits=None, other_targets=None, num_samples=5):
    """
    Compute accuracy using softmax sampling with multinomial selection.

    Optionally, a second set of logits and targets, for a second
    predicted variable, can be provided. In this case, probabilities
    associated with both sets of logits are combined to arrive at the
    best sampled predictions.

    Args:
        logits (torch.FloatTensor): Predicted logits, shape (N, vocab_size).
        targets (torch.LongTensor): Actual targets, shape (N).
        other_logits (torch.FloatTensor, optional): Predicted logits for a second variable, shape (N, other_vocab_size).
        other_targets (torch.LongTensor, optional): Actual targets for a second variable, shape (N).
        num_samples (int, optional): Number of samples to draw from the probability distribution. Defaults to 5.

    Returns:
        float: Accuracy based on softmax sampling.
    """
    with torch.no_grad():
        batch_size = logits.shape[0]

        if other_logits is not None:
            # Compute softmax probabilities
            probabilities = F.softmax(logits, dim=-1).unsqueeze(2)  # (N, vocab_size, 1)
            other_probabilities = F.softmax(other_logits, dim=-1).unsqueeze(1)  # (N, 1, other_vocab_size)

            # Compute joint probabilities
            combined_probabilities = torch.bmm(probabilities, other_probabilities).view(batch_size, -1)  # (N, vocab_size * other_vocab_size)

            # Sample from the probability distribution
            sampled_indices = torch.multinomial(combined_probabilities, num_samples, replacement=True)  # (N, num_samples)

            # Convert sampled indices back to separate predictions
            indices = sampled_indices // other_logits.shape[-1]  # (N, num_samples)
            other_indices = sampled_indices % other_logits.shape[-1]  # (N, num_samples)

            # Compare sampled values with ground truth
            correct_predictions = ((indices == targets.unsqueeze(1)) & (other_indices == other_targets.unsqueeze(1))).any(dim=1)

        else:
            # Compute softmax probabilities
            probabilities = F.softmax(logits, dim=-1)  # (N, vocab_size)

            # Sample from the probability distribution
            sampled_indices = torch.multinomial(probabilities, num_samples, replacement=True)  # (N, num_samples)

            # Compare sampled values with ground truth
            correct_predictions = (sampled_indices == targets.unsqueeze(1)).any(dim=1)

        # Compute accuracy
        accuracy = correct_predictions.float().mean().item()
        return accuracy

def topk_accuracy_single_sample(logits, targets, other_logits=None, other_targets=None, k=[1, 3, 5]):
    """
    Compute "top-k" accuracies for a single sample.

    Args:
        logits (torch.FloatTensor): Predicted logits, of size (vocab_size).
        targets (torch.LongTensor): Actual targets, of size (1).
        other_logits (torch.FloatTensor, optional): Predicted logits for a second predicted variable, size (other_vocab_size).
        other_targets (torch.LongTensor, optional): Actual targets for a second predicted variable, size (1).
        k (list, optional): Values of "k". Defaults to [1, 3, 5].

    Returns:
        list: "Top-k" accuracies.
    """
    with torch.no_grad():
        # Ensure logits are of the correct shape
        logits = logits.unsqueeze(0)  # Add batch dimension (1, vocab_size)
        targets = targets.unsqueeze(0)  # Add batch dimension (1, )

        if other_logits is not None:
            other_logits = other_logits.unsqueeze(0)  # Add batch dimension (1, other_vocab_size)
            other_targets = other_targets.unsqueeze(0)  # Add batch dimension (1, )

            # Compute top-k indices
            probabilities = F.softmax(logits, dim=-1)  # (1, vocab_size)
            other_probabilities = F.softmax(other_logits, dim=-1)  # (1, other_vocab_size)

            combined_probabilities = torch.bmm(probabilities.unsqueeze(2), other_probabilities.unsqueeze(1)).view(1, -1)  # (1, vocab_size * other_vocab_size)
            _, flattened_indices = combined_probabilities.topk(k=max(k), dim=1)  # (1, max(k))
            indices = flattened_indices // other_logits.shape[-1]  # (1, max(k))
            other_indices = flattened_indices % other_logits.shape[-1]  # (1, max(k))

            # Compare predictions with targets
            correct_predictions = (indices == targets) & (other_indices == other_targets)  # (1, max(k))

        else:
            # Compute top-k indices for single logits
            _, indices = logits.topk(k=max(k), dim=1)  # (1, max(k))

            # Compare predictions with targets
            correct_predictions = indices == targets  # (1, max(k))

        # Calculate top-k accuracies
        topk_accuracies = [
            correct_predictions[:, :k_value].sum().item() / 1 for k_value in k  # For single sample
        ]

        return topk_accuracies
