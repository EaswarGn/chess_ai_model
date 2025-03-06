import sys
from utils import *
from configs import import_config
from criteria_ddp import MultiTaskChessLoss, LabelSmoothedCE
from datasets_ddp import ChunkLoader
from model_ddp import ChessTemporalTransformerEncoder
import numpy as np
import subprocess
import random
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

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

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()
    
def calculate_accuracy(predictions, targets):
    predicted_classes = torch.argmax(predictions, dim=1)
    correct = torch.eq(predicted_classes, targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy

def validate_model(rank, world_size, CONFIG):
    setup_ddp(rank, world_size)
    
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO' 
    
    DEVICE = torch.device(f"cuda:{rank}")
    if rank == 0:
        print(f"Evaluating model on {world_size} GPU(s) with {CONFIG.NUM_WORKERS} worker(s) per GPU for dataloading.")
        
    model = ChessTemporalTransformerEncoder(CONFIG, DEVICE=DEVICE).to(DEVICE)
    
    if CONFIG.CHECKPOINT_PATH is not None: #and rank == 0:
        checkpoint = torch.load(CONFIG.CHECKPOINT_PATH, map_location=DEVICE)
        
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '')
            new_key = new_key.replace('module.', '')
            #new_key = 'module.'+new_key
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=CONFIG.USE_STRICT)
        print(f"checkpoint loaded on rank {rank}")
    else:
        print("model checkpoint path not specified, exiting...")
        cleanup_ddp()
        sys.exit()
        
    # Compile model
    compiled_model = torch.compile(
        model,
        mode=CONFIG.COMPILATION_MODE,
        dynamic=CONFIG.DYNAMIC_COMPILATION,
        disable=CONFIG.DISABLE_COMPILATION,
    )
    
    model = DDP(compiled_model, device_ids=[rank], find_unused_parameters=True)
    
    rand_folder = random.randint(1, 3)
    testing_file_list = get_all_record_files(f'../../../ranged_chunks_zipped/1900/{rand_folder}_chunks')
    testing_file_list = [file for file in testing_file_list if file.endswith('.zst')]
    testing_file_list = [s for s in testing_file_list if "._" not in s]
    testing_file_list = random.sample(testing_file_list, min(2, len(testing_file_list)))
    val_dataset = ChunkLoader(testing_file_list, record_dtype, rank, world_size, is_val=True)
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=CONFIG.BATCH_SIZE // world_size,
        num_workers=CONFIG.NUM_WORKERS, 
        pin_memory=CONFIG.PIN_MEMORY,
        prefetch_factor=CONFIG.PREFETCH_FACTOR,
    )
    
    model.eval()
    
    
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
            device=DEVICE
        )
    criterion = criterion.to(DEVICE)
    total_steps = CONFIG.VALIDATION_STEPS
    
    pbar = None
    if rank==0:
        pbar = tqdm(total=min(total_steps, len(val_loader)), desc="Validating")#, disable=(rank == 0))
    
    with torch.no_grad():
        # Batches
        for i, batch in enumerate(val_loader):
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
            if rank==0:
                pbar.update(1)
            
            if i>=total_steps:
                pbar.close()
                cleanup_ddp()
                break
            
        if rank==0:
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
            sys.exit()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name)
    CONFIG = CONFIG.CONFIG()

    world_size = CONFIG.NUM_GPUS
    mp.spawn(
        validate_model,
        args=(world_size, CONFIG),
        nprocs=world_size,
        join=True
    )