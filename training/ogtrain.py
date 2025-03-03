import time
import argparse
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import *
from configs import import_config
import numpy as np
import random
from datasets import ChunkLoader
from model import ChessTransformerEncoderFT

DEVICE = None
cudnn.benchmark = False
rating = 1900

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


def train_model(CONFIG):
    """
    Training and validation.

    Args:

        CONFIG (dict): Configuration. See ./configs.
    """
    writer = SummaryWriter(log_dir=CONFIG.LOGS_FOLDER)
    
    global DEVICE
    DEVICE = torch.device(
        f"cuda:{CONFIG.GPU_ID}" if torch.cuda.is_available() else "cpu"
    )  # CPU isn't really practical here
    print(f"training on {DEVICE} with {CONFIG.NUM_WORKERS} workers for dataloading.")

    training_file_list = get_all_record_files('../../1900_zipped_training_chunks')
    training_file_list = [file for file in training_file_list if file.endswith('.zst')]   
    training_file_list = [s for s in training_file_list if "._" not in s]
    
    rand_folder = random.randint(1, 3)
    testing_file_list = get_all_record_files(f'../../ranged_chunks_zipped/1900/{rand_folder}_chunks')
    testing_file_list = [file for file in testing_file_list if file.endswith('.zst')]
    testing_file_list = [s for s in testing_file_list if "._" not in s]
    testing_file_list = random.sample(testing_file_list, min(3, len(testing_file_list)))
    
    
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

    # Model
    model = ChessTransformerEncoderFT(CONFIG)
    model = model.to(DEVICE)

    # Optimizer
    optimizer = CONFIG.OPTIMIZER(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=CONFIG.LR,
        betas=CONFIG.BETAS,
        eps=CONFIG.EPSILON,
    )

    start_epoch = 0
    # Load checkpoint if available
    if CONFIG.TRAINING_CHECKPOINT is not None:
        checkpoint = torch.load(
            CONFIG.TRAINING_CHECKPOINT,
            weights_only=True,
        )
        #start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
    epochs = (CONFIG.N_STEPS // (len(train_loader) // CONFIG.BATCHES_PER_STEP)) + 1

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Step
        step = epoch * len(train_loader) // CONFIG.BATCHES_PER_STEP

        # One epoch's training
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
        )

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

    # Starting time
    start_data_time = time.time()
    start_step_time = time.time()

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
                predicted_moves = model(batch)  # (N, n_moves, move_vocab_size)
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
                predicted_from_squares, predicted_to_squares = model(
                    batch
                )  # (N, 1, 64), (N, 1, 64)

                # Loss
                loss = criterion(
                    predicted=predicted_from_squares,
                    targets=batch["from_squares"],
                    lengths=batch["lengths"],
                ) + criterion(
                    predicted=predicted_to_squares,
                    targets=batch["to_squares"],
                    lengths=batch["lengths"],
                )  # scalar

            # Other models
            else:
                raise NotImplementedError

            loss = loss / CONFIG.BATCHES_PER_STEP

        # Backward prop.
        scaler.scale(loss).backward()

        # Keep track of losses
        losses.update(
            loss.item() * CONFIG.BATCHES_PER_STEP, batch["lengths"].sum().item()
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
                logits=predicted_from_squares[:, 0, :],  # (N, 64)
                targets=batch["from_squares"].squeeze(1),  # (N)
                other_logits=predicted_to_squares[:, 0, :],  # (N, 64)
                other_targets=batch["to_squares"].squeeze(1),  # (N)
                k=[1, 3, 5],
            )

        else:
            raise NotImplementedError
        top1_accuracies.update(top1_accuracy, batch["lengths"].shape[0])
        top3_accuracies.update(top3_accuracy, batch["lengths"].shape[0])
        top5_accuracies.update(top5_accuracy, batch["lengths"].shape[0])

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
                    "Top-1 {top1s.val:.4f} ({top1s.avg:.4f})"
                    "Top-3 {top3s.val:.4f} ({top3s.avg:.4f})"
                    "Top-5 {top5s.val:.4f} ({top5s.avg:.4f})".format(
                        epoch + 1,
                        epochs,
                        i + 1,
                        len(train_loader),
                        step,
                        CONFIG.N_STEPS,
                        step_time=step_time,
                        data_time=data_time,
                        losses=losses,
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

    # Prohibit gradient computation explicitly
    with torch.no_grad():
        losses = AverageMeter()
        top1_accuracies = AverageMeter()  # top-1 accuracy of first move
        top3_accuracies = AverageMeter()  # top-3 accuracy of first move
        top5_accuracies = AverageMeter()  # top-5 accuracy of first move
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
                    # Forward prop.
                    predicted_from_squares, predicted_to_squares = model(
                        batch
                    )  # (N, 1, 64), (N, 1, 64)
                    

                    # Loss
                    loss = criterion(
                        predicted=predicted_from_squares,
                        targets=batch["from_squares"],
                        lengths=batch["lengths"],
                    ) + criterion(
                        predicted=predicted_to_squares,
                        targets=batch["to_squares"],
                        lengths=batch["lengths"],
                    )  # scalar

                # Other models
                else:
                    raise NotImplementedError

            # Keep track of losses
            losses.update(loss.item(), batch["lengths"].sum().item())

            # Keep track of accuracy (Direct) Move prediction models
            if CONFIG.NAME.startswith(("CT-ED-", "CT-E-")):
                top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
                    logits=predicted_moves[:, 0, :],  # (N, move_vocab_size)
                    targets=batch["moves"][:, 1],  # (N)
                    k=[1, 3, 5],
                )

            elif CONFIG.NAME.startswith(("CT-EFT-")):
                top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
                    logits=predicted_from_squares[:, 0, :],  # (N, 64)
                    targets=batch["from_squares"].squeeze(1),  # (N)
                    other_logits=predicted_to_squares[:, 0, :],  # (N, 64)
                    other_targets=batch["to_squares"].squeeze(1),  # (N)
                    k=[1, 3, 5],
                )

            else:
                raise NotImplementedError
            top1_accuracies.update(top1_accuracy, batch["lengths"].shape[0])
            top3_accuracies.update(top3_accuracy, batch["lengths"].shape[0])
            top5_accuracies.update(top5_accuracy, batch["lengths"].shape[0])

        # Log to tensorboard
        writer.add_scalar(
            tag="val/loss", scalar_value=losses.avg, global_step=epoch + 1
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