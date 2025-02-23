import time
import argparse
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import math
import torch.nn as nn

from tqdm import tqdm
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import shutil

import sys
from utils import *
from configs import import_config
from criteria import MultiTaskChessLoss
#from d import ChessDataset, ChessDatasetFT
from datasets import ChunkLoader
from model import ChessTemporalTransformerEncoder
import numpy as np
import subprocess
import random

DEVICE = None
cudnn.benchmark = False

record_dtype = np.dtype([
    ("turn", np.int8),
    ("white_kingside_castling_rights", np.int8),
    ("white_queenside_castling_rights", np.int8),
    ("black_kingside_castling_rights", np.int8),
    ("black_queenside_castling_rights", np.int8),
    ("board_position", np.int8, (64,)),
    ("from_square", np.int8),
    ("to_square", np.int8),
    ("length", np.int8),
    ("phase", np.int8),
    ("result", np.int8),
    ("categorical_result", np.int8),
    ("base_time", np.int16),
    ("increment_time", np.int16),
    ("white_remaining_time", np.float16),
    ("black_remaining_time", np.float16),
    ("white_rating", np.int16),
    ("black_rating", np.int16),
    ("time_spent_on_move", np.float16),
    ("move_number", np.int16),
    ("num_legal_moves", np.int16),
    ("white_material_value", np.int16),
    ("black_material_value", np.int16),
    ("material_difference", np.int16),
    ("moves_until_end", np.float16)
])

rating = 1900


def train_model(CONFIG):
    """
    Training and validation.

    Args:

        CONFIG (dict): Configuration. See ./configs.
    """
    global DEVICE
    DEVICE = torch.device(
        f"cuda:{CONFIG.GPU_ID}" if torch.cuda.is_available() else "cpu"
    )  # CPU isn't really practical here
    print(f"training on {DEVICE}")
    
    
    os.makedirs(f"{CONFIG.NAME}/logs/main_log", exist_ok=True)
    writer = SummaryWriter(log_dir=f'{CONFIG.NAME}/logs/main_log')
    tensorboard_process = subprocess.Popen(["tensorboard", f"--logdir={CONFIG.NAME}/logs/main_log"])
    time.sleep(5)
    
    
    # Model
    model = ChessTemporalTransformerEncoder(CONFIG).to(DEVICE)

    # Optimizer
    optimizer = CONFIG.OPTIMIZER(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=CONFIG.LR,
        #lr=0.0005,
        betas=CONFIG.BETAS,
        eps=CONFIG.EPSILON,
    )

    step = 0
    start_epoch = 0
    total_steps = CONFIG.N_STEPS
    steps_per_epoch = CONFIG.STEPS_PER_EPOCH
    epochs = total_steps//steps_per_epoch
    
    # Load checkpoint if available
    if CONFIG.CHECKPOINT_PATH is not None:
        checkpoint = torch.load(
            CONFIG.CHECKPOINT_PATH,
            weights_only=True,
        )
        
        step = checkpoint['step']
        step = int(step)
        start_epoch = step//CONFIG.STEPS_PER_EPOCH + 1
        
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '')  # remove the '_orig_mod' prefix
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print("\nLoaded checkpoint from epoch %d.\n" % start_epoch)

    # Compile model
    compiled_model = torch.compile(
        model,
        mode=CONFIG.COMPILATION_MODE,
        dynamic=CONFIG.DYNAMIC_COMPILATION,
        disable=CONFIG.DISABLE_COMPILATION,
    )

    # Loss function
    criterion = CONFIG.CRITERION(
        eps=CONFIG.LABEL_SMOOTHING, n_predictions=CONFIG.N_MOVES
    )
    criterion = criterion.to(DEVICE)

    # AMP scaler
    scaler = GradScaler(device=DEVICE, enabled=CONFIG.USE_AMP)
    
    training_file_list = get_all_record_files('~/1900_zipped_training_chunks')
    training_file_list = [file for file in training_file_list if file.endswith('.zst')]   
    training_file_list = [s for s in training_file_list if "._" not in s]
    
    rand_folder = random.randint(1, 3)
    testing_file_list = get_all_record_files(f'~/ranged_chunks_zipped/1900/{rand_folder}_chunks')
    testing_file_list = [file for file in testing_file_list if file.endswith('.zst')]
    testing_file_list = [s for s in testing_file_list if "._" not in s]
    testing_file_list = random.sample(testing_file_list, min(10, len(testing_file_list)))
    
    
    train_dataset = ChunkLoader(training_file_list, record_dtype)
    val_dataset = ChunkLoader(testing_file_list, record_dtype)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory = CONFIG.PIN_MEMORY,
        prefetch_factor=CONFIG.PREFETCH_FACTOR,
        shuffle=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory = CONFIG.PIN_MEMORY,
        prefetch_factor=CONFIG.PREFETCH_FACTOR,
        shuffle=False,
    )
    
    # One epoch's training
    train_epoch(
        train_loader=train_loader,
        val_loader=val_loader,
        model=compiled_model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        epoch=start_epoch,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        step=step,
        writer=writer,
        CONFIG=CONFIG,
    )
    
            
def calculate_accuracy(predictions, targets):
    predicted_classes = torch.argmax(predictions, dim=1)
    correct = torch.eq(predicted_classes, targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy



def train_epoch(
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    scaler,
    epoch,
    epochs,
    steps_per_epoch,
    step,
    writer,
    CONFIG,
):
    """
    One epoch's training.

    Args:

        train_loader (torch.utils.data.DataLoader): Loader for training
        data.

        model (torch.nn.Module): Model.

        criterion (torch.nn.Module): Loss criterion.

        optimizer (torch.optim.adam.Adam): Optimizer.

        scaler (torch.cuda.amp.GradScaler): AMP scaler.

        epoch (int): Epoch number.

        epochs (int): Total number of epochs.

        step (int): Step number.

        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard
        writer.

        CONFIG (dict): Configuration.
    """
    model.train()  # training mode enables dropout

    # Track some metrics
    data_time = AverageMeter()  # data loading time
    step_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss
    top1_accuracies = AverageMeter()  # top-1 accuracy of first move
    top3_accuracies = AverageMeter()  # top-3 accuracy of first move
    top5_accuracies = AverageMeter()  # top-5 accuracy of first move
    result_losses = AverageMeter()
    move_time_losses = AverageMeter()
    move_losses = AverageMeter()
    moves_until_end_losses = AverageMeter()
    categorical_game_result_losses = AverageMeter()
    categorical_game_result_accuracies = AverageMeter()

    # Starting time
    start_data_time = time.time()
    start_step_time = time.time()
    
    crossentropy_loss = nn.CrossEntropyLoss()
    
    move_loss_criterion = criterion
    criterion = MultiTaskChessLoss(
        CONFIG
    )
    criterion = criterion.to(DEVICE)

    # Batches
    for i, batch in enumerate(train_loader):
        # Move to default device
        for key in batch:
            batch[key] = batch[key].to(DEVICE)

        # Time taken to load data
        data_time.update(time.time() - start_data_time)

        with torch.autocast(
            device_type=DEVICE.type, dtype=torch.float16, enabled=CONFIG.USE_AMP
        ):
            # Forward prop.
            predictions = model(
                batch
            )  # (N, 1, 64), (N, 1, 64)
            
            loss, loss_details = criterion(predictions, batch)
            result_loss = loss_details['result_loss']
            move_time_loss = loss_details['time_loss']
            move_loss = loss_details['move_loss']
            moves_until_end_loss = loss_details['moves_until_end_loss']
            categorical_game_result_loss = loss_details['categorical_game_result_loss']

            loss = loss / CONFIG.BATCHES_PER_STEP
            result_loss = result_loss / CONFIG.BATCHES_PER_STEP
            move_time_loss = move_time_loss / CONFIG.BATCHES_PER_STEP
            move_loss = move_loss / CONFIG.BATCHES_PER_STEP
            moves_until_end_loss = moves_until_end_loss / CONFIG.BATCHES_PER_STEP
            categorical_game_result_loss = categorical_game_result_loss / CONFIG.BATCHES_PER_STEP

        if math.isnan(loss):
            sys.exit()

        # Backward prop.
        scaler.scale(loss).backward()

        # Keep track of losses
        losses.update(
            loss.item() * CONFIG.BATCHES_PER_STEP, batch["lengths"].sum().item()
        )
        result_losses.update(
            result_loss.item() * CONFIG.BATCHES_PER_STEP, batch["lengths"].sum().item()
        )
        move_time_losses.update(
            move_time_loss.item() * CONFIG.BATCHES_PER_STEP, batch["lengths"].sum().item()
        )
        move_losses.update(
            move_loss.item() * CONFIG.BATCHES_PER_STEP, batch["lengths"].sum().item()
        )
        moves_until_end_losses.update(
            moves_until_end_loss.item() * CONFIG.BATCHES_PER_STEP, batch["lengths"].sum().item()
        )
        categorical_game_result_losses.update(
            categorical_game_result_loss.item() * CONFIG.BATCHES_PER_STEP, batch["lengths"].sum().item()
        )
        
        if predictions['from_squares'] is None:
            top1_accuracy, top3_accuracy, top5_accuracy = 0.0, 0.0, 0.0
        else:
            top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
                    logits=predictions['from_squares'][:, 0, :],  # (N, 64)
                    targets=batch["from_squares"].squeeze(1),  # (N)
                    other_logits=predictions['to_squares'][:, 0, :],  # (N, 64)
                    other_targets=batch["to_squares"].squeeze(1),  # (N)
                    k=[1, 3, 5],
                )
        
        top1_accuracies.update(top1_accuracy, batch["lengths"].shape[0])
        top3_accuracies.update(top3_accuracy, batch["lengths"].shape[0])
        top5_accuracies.update(top5_accuracy, batch["lengths"].shape[0])
        
        if predictions['categorical_game_result'] is None:
            categorical_game_result_accuracies.update(0.0, batch['lengths'].shape[0])
        
        else:
            categorical_game_result_accuracies.update(calculate_accuracy(predictions['categorical_game_result'].float(),
                        batch['categorical_result']), batch["lengths"].shape[0])

        # Update model (i.e. perform a training step) only after
        # gradients are accumulated from batches_per_step batches
        if (i + 1) % CONFIG.BATCHES_PER_STEP == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # This step is now complete
            step += 1

            # Update learning rate after each step
            change_lr(
                optimizer,
                new_lr=get_lr(
                    step=step,
                    d_model=CONFIG.D_MODEL,
                    warmup_steps=CONFIG.WARMUP_STEPS,
                    schedule=CONFIG.LR_SCHEDULE,
                    decay=CONFIG.LR_DECAY,
                ),
            )

            # Time taken for this training step
            step_time.update(time.time() - start_step_time)
            
            if step % steps_per_epoch == 0:
                
                save_checkpoint(rating, step, model, optimizer, CONFIG.NAME, "checkpoints/models")
                
                # One epoch's validation
                validate_epoch(
                    val_loader=val_loader,
                    model=model,
                    criterion=move_loss_criterion,
                    epoch=epoch,
                    writer=writer,
                    CONFIG=CONFIG,
                )
                
                epoch+=1

            # Print status
            if step % CONFIG.PRINT_FREQUENCY == 0:
                print(
                    "Epoch {0}/{1}---"
                    "Batch {2}/{3}---"
                    "Step {4}/{5}---"
                    "Data Time {data_time.val:.3f} ({data_time.avg:.3f})---"
                    "Step Time {step_time.val:.3f} ({step_time.avg:.3f})---"
                    "Loss {losses.val:.4f} ({losses.avg:.4f})---"
                    "Move loss {move_losses.val:.4f} ({move_losses.avg:.4f})---"
                    "Game result loss {result_losses.val:.4f} ({result_losses.avg:.4f})---"
                    "Move time loss {move_time_losses.val:.4f} ({move_time_losses.avg:.4f})---"
                    "Move until end loss {moves_until_end_losses.val:.4f} ({moves_until_end_losses.avg:.4f})---"
                    "Categorical Game result loss {categorical_game_result_losses.val:.4f} ({categorical_game_result_losses.avg:.4f})---"
                    "Categorical Game result accuracy {categorical_game_result_accuracies.val:.4f} ({categorical_game_result_accuracies.avg:.4f})---"
                    "Top-1 {top1s.val:.4f} ({top1s.avg:.4f})"
                    "Top-3 {top3s.val:.4f} ({top3s.avg:.4f})"
                    "Top-5 {top5s.val:.4f} ({top5s.avg:.4f})".format(
                        epoch,
                        epochs,
                        i + 1,
                        len(train_loader),
                        step,
                        CONFIG.N_STEPS,
                        step_time=step_time,
                        data_time=data_time,
                        losses=losses,
                        move_losses=move_losses,
                        result_losses=result_losses,
                        move_time_losses=move_time_losses,
                        moves_until_end_losses=moves_until_end_losses,
                        categorical_game_result_losses=categorical_game_result_losses,
                        categorical_game_result_accuracies=categorical_game_result_accuracies,
                        top1s=top1_accuracies,
                        top3s=top3_accuracies,
                        top5s=top5_accuracies,
                    )
                )

            # Log to tensorboard
            writer.add_scalar(
                tag="train/loss", scalar_value=losses.val, global_step=step
            )
            writer.add_scalar(
                tag="train/move_loss", scalar_value=move_losses.val, global_step=step
            )
            writer.add_scalar(
                tag="train/result_loss", scalar_value=result_losses.val, global_step=step
            )
            writer.add_scalar(
                tag="train/move_time_loss", scalar_value=move_time_losses.val, global_step=step
            )
            writer.add_scalar(
                tag="train/moves_until_end_loss", scalar_value=moves_until_end_losses.val, global_step=step
            )
            writer.add_scalar(
                tag="train/categorical_game_result_loss", scalar_value=categorical_game_result_losses.val, global_step=step
            )
            writer.add_scalar(
                tag="train/categorical_game_result_accuracy", scalar_value=categorical_game_result_accuracies.val, global_step=step
            )
            writer.add_scalar(
                tag="train/lr",
                scalar_value=optimizer.param_groups[0]["lr"],
                global_step=step,
            )
            writer.add_scalar(
                tag="train/data_time", scalar_value=data_time.val, global_step=step
            )
            writer.add_scalar(
                tag="train/step_time", scalar_value=step_time.val, global_step=step
            )
            writer.add_scalar(
                tag="train/top1_accuracy",
                scalar_value=top1_accuracies.val,
                global_step=step,
            )
            writer.add_scalar(
                tag="train/top3_accuracy",
                scalar_value=top3_accuracies.val,
                global_step=step,
            )
            writer.add_scalar(
                tag="train/top5_accuracy",
                scalar_value=top5_accuracies.val,
                global_step=step,
            )
            
            if step >= CONFIG.N_STEPS:
                sys.exit()

            # Reset step time
            start_step_time = time.time()

        # Reset data time
        start_data_time = time.time()

def validate_epoch(val_loader, model, criterion, epoch, writer, CONFIG):
    """
    One epoch's validation.

    Args:

        val_loader (torch.utils.data.DataLoader): Loader for validation
        data

        model (torch.nn.Module): Model

        criterion (torch.nn.Module): Loss criterion.

        epoch (int): Epoch number.

        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard
        writer.

        CONFIG (dict): Configuration.
    """
    print("\n")
    model.eval()  # eval mode disables dropout
    
    losses = AverageMeter()  # loss
    top1_accuracies = AverageMeter()  # top-1 accuracy of first move
    top3_accuracies = AverageMeter()  # top-3 accuracy of first move
    top5_accuracies = AverageMeter()  # top-5 accuracy of first move
    result_losses = AverageMeter()
    move_time_losses = AverageMeter()
    move_losses = AverageMeter()
    moves_until_end_losses = AverageMeter()
    categorical_game_result_losses = AverageMeter()
    categorical_game_result_accuracies = AverageMeter()
    
    crossentropy_loss = nn.CrossEntropyLoss()
    
    criterion = MultiTaskChessLoss(
        CONFIG
    )
    criterion = criterion.to(DEVICE)

    # Prohibit gradient computation explicitly
    with torch.no_grad():
        # Batches
        for i, batch in tqdm(
            enumerate(val_loader), desc="Validating", total=len(val_loader)
        ):
            # Move to default device
            for key in batch:
                batch[key] = batch[key].to(DEVICE)

            with torch.autocast(
                device_type=DEVICE.type, dtype=torch.float16, enabled=CONFIG.USE_AMP
            ):
                
                predictions = model(
                    batch
                )  # (N, 1, 64), (N, 1, 64)
                
                loss, loss_details = criterion(predictions, batch)
                result_loss = loss_details['result_loss']
                move_time_loss = loss_details['time_loss']
                move_loss = loss_details['move_loss']
                moves_until_end_loss = loss_details['moves_until_end_loss']
                categorical_game_result_loss = loss_details['categorical_game_result_loss']

            losses.update(
                loss.item(), batch["lengths"].sum().item()
            )
            result_losses.update(
                result_loss.item(), batch["lengths"].sum().item()
            )
            move_time_losses.update(
                move_time_loss.item(), batch["lengths"].sum().item()
            )
            move_losses.update(
                move_loss.item(), batch["lengths"].sum().item()
            )
            moves_until_end_losses.update(
                moves_until_end_loss.item(), batch["lengths"].sum().item()
            )
            categorical_game_result_losses.update(
                categorical_game_result_loss.item(), batch["lengths"].sum().item()
            )

            if predictions['from_squares'] is None:
                top1_accuracy, top3_accuracy, top5_accuracy = 0.0, 0.0, 0.0
            else:
                top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
                        logits=predictions['from_squares'][:, 0, :],  # (N, 64)
                        targets=batch["from_squares"].squeeze(1),  # (N)
                        other_logits=predictions['to_squares'][:, 0, :],  # (N, 64)
                        other_targets=batch["to_squares"].squeeze(1),  # (N)
                        k=[1, 3, 5],
                    )
            
            top1_accuracies.update(top1_accuracy, batch["lengths"].shape[0])
            top3_accuracies.update(top3_accuracy, batch["lengths"].shape[0])
            top5_accuracies.update(top5_accuracy, batch["lengths"].shape[0])
            
            if predictions['categorical_game_result'] is None:
                categorical_game_result_accuracies.update(0.0, batch['lengths'].shape[0])
            else:
                categorical_game_result_accuracies.update(calculate_accuracy(predictions['categorical_game_result'].float(),
                            batch['categorical_result']), batch["lengths"].shape[0])

        # Log to tensorboard
        writer.add_scalar(
            tag="val/loss", scalar_value=losses.avg, global_step=epoch + 1
        )
        writer.add_scalar(
            tag="val/move_loss", scalar_value=move_losses.avg, global_step=epoch + 1
        )
        writer.add_scalar(
            tag="val/result_loss", scalar_value=result_losses.avg, global_step=epoch + 1
        )
        writer.add_scalar(
            tag="val/move_time_loss", scalar_value=move_time_losses.avg, global_step=epoch + 1
        )
        writer.add_scalar(
            tag="val/moves_until_end_loss", scalar_value=moves_until_end_losses.avg, global_step=epoch + 1
        )
        writer.add_scalar(
                tag="val/categorical_game_result_loss", scalar_value=categorical_game_result_losses.avg, global_step=epoch+1
            )
        writer.add_scalar(
            tag="val/categorical_game_result_accuracy", scalar_value=categorical_game_result_accuracies.avg, global_step=epoch+1
        )
        writer.add_scalar(
            tag="val/top1_accuracy",
            scalar_value=top1_accuracies.avg,
            global_step=epoch + 1,
        )
        writer.add_scalar(
            tag="val/top3_accuracy",
            scalar_value=top3_accuracies.avg,
            global_step=epoch + 1,
        )
        writer.add_scalar(
            tag="val/top5_accuracy",
            scalar_value=top5_accuracies.avg,
            global_step=epoch + 1,
        )

        print("\nValidation loss: %.3f" % losses.avg)
        print("\nValidation move loss: %.3f" % move_losses.avg)
        print("\nValidation result loss: %.3f" % result_losses.avg)
        print("\nValidation move time loss: %.3f" % move_time_losses.avg)
        print("\nValidation moves until end loss: %.3f" % moves_until_end_losses.avg)
        print("\nValidation Categorical game result loss: %.3f" % categorical_game_result_losses.avg)
        print("\nValidation Categorical game result accuracy: %.3f" % categorical_game_result_accuracies.avg)
        print("Validation top-1 accuracy: %.3f" % top1_accuracies.avg)
        print("Validation top-3 accuracy: %.3f" % top3_accuracies.avg)
        print("Validation top-5 accuracy: %.3f\n" % top5_accuracies.avg)


if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name)

    # Train model
    train_model(CONFIG)