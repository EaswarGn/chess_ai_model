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
from datasets import ChessDatasetFT
from model import ChessTemporalTransformerEncoder

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # CPU isn't really practical here
cudnn.benchmark = False


def train_model(CONFIG):
    """
    Training and validation.

    Args:

        CONFIG (dict): Configuration. See ./configs.
    """
    writer = SummaryWriter(log_dir='logs')

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
    
    """checkpoint_path = ''
    if DEVICE.type == 'cpu':
        checkpoint_path = 'checkpoints/CT-EFT-85.pt'
    else:
        checkpoint_path = '../../../../drive/My Drive/CT-EFT-85.pt'
    checkpoint = torch.load(str(checkpoint_path), weights_only=True)
    state_dict = checkpoint['model_state_dict']
    
    # Strip the _orig_mod prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '')  # remove the '_orig_mod' prefix
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])"""

    # Load checkpoint if available
    if CONFIG.TRAINING_CHECKPOINT is not None:
        checkpoint = torch.load(
            os.path.join(CONFIG.CHECKPOINT_FOLDER, CONFIG.TRAINING_CHECKPOINT),
            weights_only=True,
        )
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("\nLoaded checkpoint from epoch %d.\n" % start_epoch)
    else:
        start_epoch = 0

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

    # Find total epochs to train
    #epochs = (CONFIG.N_STEPS // (len(train_loader) // CONFIG.BATCHES_PER_STEP)) + 1
    epochs = 9
    num_epochs_completed = 0
    num_full_loops = 0
    start_epoch = 0

    # Epochs
    for epoch in range(start_epoch, epochs):
        #epoch = epoch+1
        
        train_data_folder = ''
        val_data_folder = ''
        
        data_epoch = (epoch%3)+1
        if DEVICE.type == 'cpu':
            train_data_folder = f'../data/epoch_{data_epoch}/train_data'
            val_data_folder = f'../data/epoch_{data_epoch}/val_data'
        else:
            train_data_folder = f'../../drive/My Drive/data/epoch_{data_epoch}/train_data'
            val_data_folder = f'../../drive/My Drive/data/epoch_{data_epoch}/val_data'
            
        
        train_loader = DataLoader(
            dataset=ChessDatasetFT(
                data_folder=train_data_folder,
                h5_file='data.h5',
                split="train",
                n_moves=CONFIG.N_MOVES,
            ),
            batch_size=CONFIG.BATCH_SIZE,
            num_workers=CONFIG.NUM_WORKERS,
            pin_memory=CONFIG.PIN_MEMORY,
            prefetch_factor=CONFIG.PREFETCH_FACTOR,
            shuffle=True,
        )
        val_loader = DataLoader(
            dataset=ChessDatasetFT(
                data_folder=val_data_folder,
                h5_file='data.h5',
                split="val",
                n_moves=CONFIG.N_MOVES,
            ),
            batch_size=CONFIG.BATCH_SIZE,
            num_workers=CONFIG.NUM_WORKERS,
            pin_memory=CONFIG.PIN_MEMORY,
            prefetch_factor=CONFIG.PREFETCH_FACTOR,
            shuffle=False,
        )
        
        #epoch = epoch - 1

        # Step
        step = epoch * len(train_loader) // CONFIG.BATCHES_PER_STEP

        """# One epoch's training
        train_epoch(
            train_loader=train_loader,
            model=compiled_model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            epochs=epochs,
            step=step,
            writer=writer,
            CONFIG=CONFIG,
        )"""

        # One epoch's validation
        validate_epoch(
            val_loader=val_loader,
            model=compiled_model,
            criterion=criterion,
            epoch=epoch,
            writer=writer,
            CONFIG=CONFIG,
        )

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, CONFIG.NAME, CONFIG.CHECKPOINT_FOLDER)
        
        num_epochs_completed += 1
        
        if num_epochs_completed%3==0:
            num_full_loops += 1
            model_file = 'checkpoints/CT-EFT-85/CT-EFT-85.pt'

            # Destination path in Google Drive (choose your own folder and filename)
            destination = f'../../drive/My Drive/CT-EFT-85_{num_full_loops}.pt'
            shutil.copy(model_file, destination)
            
def calculate_accuracy(predictions, targets):
    predicted_classes = torch.argmax(predictions, dim=1)
    correct = torch.eq(predicted_classes, targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy



def train_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    scaler,
    epoch,
    epochs,
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
    
    criterion = MultiTaskChessLoss(
        move_weight=1.0, 
        time_weight=0.5, 
        result_weight=1.0,
        moves_until_end_weight=0.5,
        temperature=1.0,
        criterion = criterion
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
            # (Direct) Move prediction models
            if CONFIG.NAME.startswith(("CT-ED-", "CT-E-")):
                # Forward prop.
                #predicted_moves = model(batch)  # (N, n_moves, move_vocab_size)
                predicted_moves = model(batch)
                # Note: n_moves is how many moves into the future we are
                # targeting for modeling. For an Encoder-Decoder model,
                # this might be max_move_sequence_length. For an
                # Encoder-only model, this will be 1.

                # Loss
                loss = criterion(
                    predicted=predicted_moves,  # (N, n_moves, move_vocab_size)
                    targets=batch["moves"][:, 1:],  # (N, n_moves)
                    lengths=batch["lengths"],  # (N, 1)
                )  # scalar
                # Note: We don't pass the first move (the prompt
                # "<move>") as it is not a target/next-move of anything

            # "From" and "To" square prediction models
            elif CONFIG.NAME.startswith(("CT-EFT-")):
                # Forward prop.
                predictions = model(
                    batch
                )  # (N, 1, 64), (N, 1, 64)
                
                loss, loss_details = criterion(predictions, batch)
                result_loss = loss_details['result_loss']
                move_time_loss = loss_details['time_loss']
                move_loss = loss_details['move_loss']
                moves_until_end_loss = loss_details['moves_until_end_loss']
                
                batch['categorical_result'] = batch['categorical_result'].squeeze(1)
                categorical_game_result_loss = crossentropy_loss(
                    predictions['categorical_game_result'].float(),
                    batch['categorical_result']
                )

            # Other models
            else:
                raise NotImplementedError

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

        # Keep track of accuracy (Direct) Move prediction models
        if CONFIG.NAME.startswith(("CT-ED-", "CT-E-")):
            top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
                logits=predicted_moves[:, 0, :],  # (N, move_vocab_size)
                targets=batch["moves"][:, 1],  # (N)
                k=[1, 3, 5],
            )

        elif CONFIG.NAME.startswith(("CT-EFT-")):
            top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
                logits=predictions['from_squares'][:, 0, :],  # (N, 64)
                targets=batch["from_squares"].squeeze(1),  # (N)
                other_logits=predictions['to_squares'][:, 0, :],  # (N, 64)
                other_targets=batch["to_squares"].squeeze(1),  # (N)
                k=[1, 3, 5],
            )

        else:
            raise NotImplementedError
        top1_accuracies.update(top1_accuracy, batch["lengths"].shape[0])
        top3_accuracies.update(top3_accuracy, batch["lengths"].shape[0])
        top5_accuracies.update(top5_accuracy, batch["lengths"].shape[0])
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
                    "Categorical Game result accuracy {categorical_game_result_accuracies.val:.4f} ({categorical_game_result_losses.avg:.4f})---"
                    "Top-1 {top1s.val:.4f} ({top1s.avg:.4f})"
                    "Top-3 {top3s.val:.4f} ({top3s.avg:.4f})"
                    "Top-5 {top5s.val:.4f} ({top5s.avg:.4f})".format(
                        epoch,
                        epochs,
                        i + 1,
                        len(train_loader),
                        step,
                        (len(train_loader)//CONFIG.BATCHES_PER_STEP)*(epoch+1),
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

            # Reset step time
            start_step_time = time.time()

            # If this is the last one or two epochs, save checkpoints at
            # regular intervals for averaging
            if (
                epoch in [epochs - 1, epochs - 2] and step % 1500 == 0
            ):  # 'epoch' is 0-indexed
                save_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    CONFIG.NAME,
                    CONFIG.CHECKPOINT_FOLDER,
                    prefix="step" + str(step) + "_",
                )

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
        move_weight=1.0, 
        time_weight=0.5, 
        result_weight=1.0,
        moves_until_end_weight=0.5,
        temperature=1.0,
        criterion = criterion
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
                # (Direct) Move prediction models
                if CONFIG.NAME.startswith(("CT-ED-", "CT-E-")):
                    # Forward prop.
                    predicted_moves = model(batch)  # (N, n_moves, move_vocab_size)
                    # Note: n_moves is how many moves into the future we
                    # are targeting for modeling. For an Encoder-Decoder
                    # model, this might be max_move_sequence_length. For
                    # an Encoder-only model, this will be 1.

                    # Loss
                    loss = criterion(
                        predicted=predicted_moves,  # (N, n_moves, move_vocab_size)
                        targets=batch["moves"][:, 1:],  # (N, n_moves)
                        lengths=batch["lengths"],  # (N, 1)
                    )  # scalar
                    # Note: We don't pass the first move (the prompt
                    # "<move>") as it is not a target/next-move of
                    # anything

                # "From" and "To" square prediction models
                elif CONFIG.NAME.startswith(("CT-EFT-")):
                    predictions = model(
                        batch
                    )  # (N, 1, 64), (N, 1, 64)
                    
                    loss, loss_details = criterion(predictions, batch)
                    result_loss = loss_details['result_loss']
                    move_time_loss = loss_details['time_loss']
                    move_loss = loss_details['move_loss']
                    moves_until_end_loss = loss_details['moves_until_end_loss']
                    
                    batch['categorical_result'] = batch['categorical_result'].squeeze(1)
                    categorical_game_result_loss = crossentropy_loss(
                        predictions['categorical_game_result'].float(),
                        batch['categorical_result']
                    )

                # Other models
                else:
                    raise NotImplementedError

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

            # Keep track of accuracy (Direct) Move prediction models
            if CONFIG.NAME.startswith(("CT-ED-", "CT-E-")):
                top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
                    logits=predicted_moves[:, 0, :],  # (N, move_vocab_size)
                    targets=batch["moves"][:, 1],  # (N)
                    k=[1, 3, 5],
                )

            elif CONFIG.NAME.startswith(("CT-EFT-")):
                top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
                    logits=predictions['from_squares'][:, 0, :],  # (N, 64)
                    targets=batch["from_squares"].squeeze(1),  # (N)
                    other_logits=predictions['to_squares'][:, 0, :],  # (N, 64)
                    other_targets=batch["to_squares"].squeeze(1),  # (N)
                    k=[1, 3, 5],
                )

            else:
                raise NotImplementedError
            top1_accuracies.update(top1_accuracy, batch["lengths"].shape[0])
            top3_accuracies.update(top3_accuracy, batch["lengths"].shape[0])
            top5_accuracies.update(top5_accuracy, batch["lengths"].shape[0])
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
                tag="val/categorical_game_result_loss", scalar_value=categorical_game_result_losses.val, global_step=epoch+1
            )
        writer.add_scalar(
            tag="val/categorical_game_result_accuracy", scalar_value=categorical_game_result_accuracies.val, global_step=epoch+1
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