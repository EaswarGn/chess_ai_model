import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import time
import argparse
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import math
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import shutil

import sys
from utils import *
from configs import import_config
from criteria import MultiTaskChessLoss, LabelSmoothedCE
from datasets import ChunkLoader
from model import ChessTemporalTransformerEncoder, PonderingTimeModel
import numpy as np
import subprocess
import random
import datetime


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

def seed_everything(seed: int = 42):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch random seed (CPU)
    
    # If using CUDA
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  # For multi-GPU training
    
    # Ensures deterministic behavior
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  # Disables auto-optimization
    
    # Ensures deterministic behavior for `torch.use_deterministic_algorithms`
    torch.use_deterministic_algorithms(True, warn_only=True)

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    #dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, init_method='env://', timeout=datetime.timedelta(seconds=18000))

def cleanup_ddp():
    dist.destroy_process_group()

def train_model_ddp(rank, world_size, CONFIG):
    
    seed_everything(42)
    setup_ddp(rank, world_size)
    
    
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO' 
    
    DEVICE = torch.device(f"cuda:{rank}")
    if rank == 0:
        print(f"Training {CONFIG.NAME} model on {world_size} GPU(s) with {CONFIG.NUM_WORKERS} worker(s) per GPU for dataloading.")
        os.makedirs(f"{CONFIG.NAME}/logs/main_log", exist_ok=True)
        writer = SummaryWriter(log_dir=f'{CONFIG.NAME}/logs/main_log')
        print(f"TensorBoard logdir created at: {CONFIG.NAME}/logs/main_log")
        
    else:
        writer = None

    # Model
    model = None
    if "time" in CONFIG.NAME:
        model = PonderingTimeModel(CONFIG, DEVICE=DEVICE).to(DEVICE)
    else:
        model = ChessTemporalTransformerEncoder(CONFIG, DEVICE=DEVICE).to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=CONFIG.LR,
        betas=CONFIG.BETAS,
        eps=CONFIG.EPSILON,
        weight_decay=CONFIG.WEIGHT_DECAY
    )

    step = CONFIG.STEP
    start_epoch = 0
    total_steps = CONFIG.N_STEPS
    steps_per_epoch = CONFIG.STEPS_PER_EPOCH
    epochs = 50
    
    
    if CONFIG.CHECKPOINT_PATH is not None: #and rank == 0:
        checkpoint = torch.load(CONFIG.CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        
        try:
            step = checkpoint['step']
            step = int(step)
        except KeyError:
            print("step is not specified in state dict")
            step = 1
        start_epoch = step//CONFIG.STEPS_PER_EPOCH + 1
        
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '').replace('module.', '')
            new_state_dict[new_key] = value  # Copy all other weights normally
        model.load_state_dict(new_state_dict, strict=CONFIG.USE_STRICT)
        
        
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (ValueError, KeyError) as e:
            error_message = str(e)
            print("WARNING: optimizer state dict not loaded likely because you are finetuning model with different weights, but proceed with caution")
            print(f"Error Message: {error_message}")

        print(f"\nLoaded checkpoint from step {step}.\n")
    
    if CONFIG.STEP is None:
        step = 1
    
    # Compile model
    compiled_model = torch.compile(
        model,
        mode=CONFIG.COMPILATION_MODE,
        dynamic=CONFIG.DYNAMIC_COMPILATION,
        disable=CONFIG.DISABLE_COMPILATION,
    )

        
    model = DDP(compiled_model, device_ids=[rank], find_unused_parameters=True)

    criterion = LabelSmoothedCE(DEVICE=DEVICE, eps=CONFIG.LABEL_SMOOTHING, n_predictions=CONFIG.N_MOVES).to(DEVICE)
    scaler = GradScaler(enabled=CONFIG.USE_AMP)
    
    training_file_list = get_all_record_files('../../blitzking45_train_chunks') 
    training_file_list = [file for file in training_file_list if file.endswith('.zst')]   
    training_file_list = [s for s in training_file_list if "._" not in s]
    random.shuffle(training_file_list)
    
    
    testing_file_list = get_all_record_files(f'../../blitzking45_validation_chunks')
    testing_file_list = [file for file in testing_file_list if file.endswith('.zst')]
    testing_file_list = [s for s in testing_file_list if "._" not in s]
    
    use_low_time = False
    min_full_move_number = 5
    if "time" in CONFIG.NAME:
        use_low_time = True
        min_full_move_number = -1
    train_dataset = ChunkLoader(training_file_list, record_dtype, rank, world_size, include_low_time_moves=use_low_time, min_full_move_number=min_full_move_number, target_player=CONFIG.TARGET_PLAYER, loop_forever=True)
    val_dataset = ChunkLoader(testing_file_list, record_dtype, rank, world_size, include_low_time_moves=use_low_time, min_full_move_number=min_full_move_number, target_player=CONFIG.TARGET_PLAYER, loop_forever=False)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=CONFIG.BATCH_SIZE // world_size,
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory=CONFIG.PIN_MEMORY,
        prefetch_factor=CONFIG.PREFETCH_FACTOR,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=CONFIG.BATCH_SIZE // world_size,
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory=CONFIG.PIN_MEMORY,
        prefetch_factor=CONFIG.PREFETCH_FACTOR,
    )

    train_epoch(
        rank=rank,
        world_size=world_size,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        epoch=start_epoch,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        step=step,
        writer=writer,
        CONFIG=CONFIG,
        device=DEVICE
    )
    
    """#validation only
    if rank==0:
        validate_epoch(
            rank=rank,
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            epoch=0,
            writer=writer,
            CONFIG=CONFIG,
            device=DEVICE
        )
        cleanup_ddp()
        sys.exit()"""

    cleanup_ddp()

def calculate_accuracy(predictions, targets):
    predicted_classes = torch.argmax(predictions, dim=1)
    correct = torch.eq(predicted_classes, targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy

def train_epoch(
    rank,
    world_size,
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
    device
):
    model.train()

    data_time = AverageMeter()
    step_time = AverageMeter()
    losses = AverageMeter()
    top1_accuracies = AverageMeter()
    top3_accuracies = AverageMeter()
    top5_accuracies = AverageMeter()
    result_losses = AverageMeter()
    move_time_losses = AverageMeter()
    move_losses = AverageMeter()
    moves_until_end_losses = AverageMeter()
    categorical_game_result_losses = AverageMeter()
    categorical_game_result_accuracies = AverageMeter()

    start_data_time = time.time()
    start_step_time = time.time()
    
    move_loss_criterion = criterion
    criterion = MultiTaskChessLoss(CONFIG, device=device).to(device)
    
    print("no")
    for i, batch in enumerate(train_loader):
        for key in batch:
            batch[key] = batch[key].to(device)
        print("m")

        data_time.update(time.time() - start_data_time)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=CONFIG.USE_AMP):
            predictions = model(batch)
            
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
        
        scaler.scale(loss).backward()

        losses.update(loss.item() * CONFIG.BATCHES_PER_STEP, batch["lengths"].sum().item())
        result_losses.update(result_loss.item() * CONFIG.BATCHES_PER_STEP, batch["lengths"].sum().item())
        move_time_losses.update(move_time_loss.item() * CONFIG.BATCHES_PER_STEP, batch["lengths"].sum().item())
        move_losses.update(move_loss.item() * CONFIG.BATCHES_PER_STEP, batch["lengths"].sum().item())
        moves_until_end_losses.update(moves_until_end_loss.item() * CONFIG.BATCHES_PER_STEP, batch["lengths"].sum().item())
        categorical_game_result_losses.update(categorical_game_result_loss.item() * CONFIG.BATCHES_PER_STEP, batch["lengths"].sum().item())
        
        if predictions['from_squares'] is None:
            top1_accuracy, top3_accuracy, top5_accuracy = 0.0, 0.0, 0.0
        else:
            top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
                    logits=predictions['from_squares'][:, 0, :],
                    targets=batch["from_squares"].squeeze(1),
                    other_logits=predictions['to_squares'][:, 0, :],
                    other_targets=batch["to_squares"].squeeze(1),
                    k=[1, 3, 5],
                )
        
        top1_accuracies.update(top1_accuracy, batch["lengths"].shape[0])
        top3_accuracies.update(top3_accuracy, batch["lengths"].shape[0])
        top5_accuracies.update(top5_accuracy, batch["lengths"].shape[0])
        
        if predictions['categorical_game_result'] is None:
            categorical_game_result_accuracies.update(0.0, batch['lengths'].shape[0])
        else:
            categorical_game_result_accuracies.update(
                calculate_accuracy(
                    predictions['categorical_game_result'].float(),
                    batch['categorical_result']
                ),
                batch["lengths"].shape[0]
            )

        if (i + 1) % CONFIG.BATCHES_PER_STEP == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            step += 1
            
            
            
            
            change_lr(
                optimizer,
                new_lr=get_lr(
                    step=step,
                    d_model=CONFIG.D_MODEL,
                    warmup_steps=CONFIG.WARMUP_STEPS,
                    total_steps=CONFIG.N_STEPS,
                    schedule=CONFIG.LR_SCHEDULE,
                    decay=CONFIG.LR_DECAY,
                    batch_size=CONFIG.BATCH_SIZE
                ),
            )

            step_time.update(time.time() - start_step_time)

            if step % CONFIG.PRINT_FREQUENCY == 0 and rank == 0:
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
                    "Moves until end loss {moves_until_end_losses.val:.4f} ({moves_until_end_losses.avg:.4f})---"
                    "Categorical Game result loss {categorical_game_result_losses.val:.4f} ({categorical_game_result_losses.avg:.4f})---"
                    "Categorical Game result accuracy {categorical_game_result_accuracies.val:.4f} ({categorical_game_result_accuracies.avg:.4f})---"
                    "Top-1 {top1s.val:.4f} ({top1s.avg:.4f})"
                    "Top-3 {top3s.val:.4f} ({top3s.avg:.4f})"
                    "Top-5 {top5s.val:.4f} ({top5s.avg:.4f})".format(
                        epoch,
                        epochs,
                        i + 1,
                        CONFIG.N_STEPS*4,
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
                print()

            if rank == 0:
                writer.add_scalar(tag="train/loss", scalar_value=losses.val, global_step=step)
                writer.add_scalar(tag="train/move_loss", scalar_value=move_losses.val, global_step=step)
                writer.add_scalar(tag="train/result_loss", scalar_value=result_losses.val, global_step=step)
                writer.add_scalar(tag="train/move_time_loss", scalar_value=move_time_losses.val, global_step=step)
                writer.add_scalar(tag="train/moves_until_end_loss", scalar_value=moves_until_end_losses.val, global_step=step)
                writer.add_scalar(tag="train/categorical_game_result_loss", scalar_value=categorical_game_result_losses.val, global_step=step)
                writer.add_scalar(tag="train/categorical_game_result_accuracy", scalar_value=categorical_game_result_accuracies.val, global_step=step)
                writer.add_scalar(tag="train/lr", scalar_value=optimizer.param_groups[0]["lr"], global_step=step)
                writer.add_scalar(tag="train/data_time", scalar_value=data_time.val, global_step=step)
                writer.add_scalar(tag="train/step_time", scalar_value=step_time.val, global_step=step)
                writer.add_scalar(tag="train/top1_accuracy", scalar_value=top1_accuracies.val, global_step=step)
                writer.add_scalar(tag="train/top3_accuracy", scalar_value=top3_accuracies.val, global_step=step)
                writer.add_scalar(tag="train/top5_accuracy", scalar_value=top5_accuracies.val, global_step=step)
            
            if step % steps_per_epoch == 0:
                
                if rank == 0: 
                    time.sleep(5)
                    save_checkpoint(rating, step, model.module, optimizer, CONFIG.NAME, "checkpoints/models", CONFIG)
                    
                    validate_epoch(
                        rank=rank,
                        val_loader=val_loader,
                        model=model,
                        criterion=move_loss_criterion,
                        epoch=epoch,
                        writer=writer,
                        CONFIG=CONFIG,
                        device=device
                    )
                
                epoch += 1
            
            
            if CONFIG.N_STEPS is None:
                if step >= len(train_loader)//CONFIG.BATCHES_PER_STEP and rank==0:
                    save_checkpoint(rating, step, model.module, optimizer, CONFIG.NAME, "checkpoints/models", CONFIG)
                    cleanup_ddp()
                    sys.exit()
            else:
                if step >= CONFIG.N_STEPS and rank==0:
                    save_checkpoint(rating, step, model.module, optimizer, CONFIG.NAME, "checkpoints/models", CONFIG)
                    cleanup_ddp()
                    sys.exit()

            start_step_time = time.time()

        start_data_time = time.time()
     
     
#def validate_epoch(rank, val_loader, model, criterion, epoch, writer, CONFIG, device):   
def validate_epoch(rank, val_loader, model, criterion, epoch, writer, CONFIG, device):
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
    
    if rank==0:
        print("\n")
        model.eval()  # eval mode disables dropout
        
        DEVICE = device
        
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
        
        criterion = MultiTaskChessLoss(
            CONFIG,
            device=device
        )
        criterion = criterion.to(DEVICE)
        total_steps = CONFIG.VALIDATION_STEPS

        # Prohibit gradient computation explicitly
        with torch.no_grad():
            # Batches
            for i, batch in tqdm(
                enumerate(val_loader), desc="Validating", total=min(total_steps, len(val_loader))
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
                
                if i>=total_steps:
                    break
                
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name)
    CONFIG = CONFIG.CONFIG()
    
    #from picklableconfig import convert_config_to_picklable
    #CONFIG = convert_config_to_picklable(CONFIG)
    
    world_size = CONFIG.NUM_GPUS
    mp.spawn(
        train_model_ddp,
        args=(world_size, CONFIG),
        nprocs=world_size,
        join=True
    )