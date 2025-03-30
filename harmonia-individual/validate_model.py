import sys
from utils import *
from configs import import_config
from criteria import MultiTaskChessLoss, LabelSmoothedCE
from datasets import ChunkLoader
from model import ChessTemporalTransformerEncoder, PonderingTimeModel
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
from configs.models.utils.levels import SQUARES

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

range_values = [round(x, 2) for x in [i * 0.02 for i in range(51)]]

def validate_model(rank, world_size, CONFIG):
    setup_ddp(rank, world_size)
    
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO' 
    
    DEVICE = torch.device(f"cuda:{rank}")
    if rank == 0:
        print(f"Evaluating {CONFIG.NAME} model on {world_size} GPU(s) with {CONFIG.NUM_WORKERS} worker(s) per GPU for dataloading.")
        
    
    # Model
    model = None
    if "time" in CONFIG.NAME:
        model = PonderingTimeModel(CONFIG, DEVICE=DEVICE).to(DEVICE)
    else:
        model = ChessTemporalTransformerEncoder(CONFIG, DEVICE=DEVICE).to(DEVICE)
    
    if CONFIG.CHECKPOINT_PATH is not None: #and rank == 0:
        checkpoint = torch.load(CONFIG.CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '')
            new_key = new_key.replace('module.', '')
            #new_key = 'module.'+new_key
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict)
        
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
    
    testing_file_list = get_all_record_files(f'../../{CONFIG.TARGET_PLAYER}_validation_chunks')
    testing_file_list = [file for file in testing_file_list if file.endswith('.zst')]
    testing_file_list = [s for s in testing_file_list if "._" not in s]
    
    use_low_time = False
    min_full_move_number = 5
    max_full_move_number = 1e10
    if "time" in CONFIG.NAME:
        use_low_time = True
        min_full_move_number = 5
    if "opening" in CONFIG.NAME:
        use_low_time = False
        min_full_move_number = -1
        max_full_move_number = 5
    val_dataset = ChunkLoader(
        testing_file_list,
        record_dtype,
        rank,
        world_size,
        include_low_time_moves=use_low_time,
        min_full_move_number=min_full_move_number,
        max_full_move_number=max_full_move_number,
        target_player=CONFIG.TARGET_PLAYER,
        loop_forever=False
    )
    
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
    total_steps = min(CONFIG.VALIDATION_STEPS, len(val_loader))
    
    pbar = None
    if rank==0:
        pbar = tqdm(total=total_steps, desc="Validating")
    
    total_batches_processed = 0
    
    s = 0
    correct = 0
    total = 0
    softmaxsampling_accuracies = AverageMeter()
    
    with torch.no_grad():
        # Batches
        while s<len(range_values)-1:
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
                    
                    # Get the probabilities from logits
                    from_probs = torch.nn.functional.softmax(predictions['from_squares'][:, 0, :], dim=-1)
                    to_probs = torch.nn.functional.softmax(predictions['to_squares'][:, 0, :], dim=-1)

                    # Get the top 5 probabilities and their indices for each sample
                    top5_from_probs, top5_from_indices = torch.topk(from_probs, k=5, dim=-1)
                    top5_to_probs, top5_to_indices = torch.topk(to_probs, k=5, dim=-1)

                    # Get the top-1 move (the move with the highest probability) for each sample
                    top1_from_prob, top1_from_index = from_probs.max(dim=-1)
                    top1_to_prob, top1_to_index = to_probs.max(dim=-1)

                    # Compute the difference between the top-1 move and the fifth most likely move for each sample
                    prob_diff = top1_from_prob*top1_to_prob - top5_from_probs[:, -1]*top5_to_probs[:, -1]

                    # Initialize list for sampling accuracy
                    sampling_accuracy_list = []

                    # Iterate over each sample in the batch
                    for i in range(from_probs.size(0)):
                        # Check if the difference is less than 0.4 for the current sample
                        if prob_diff[i] < range_values[s]:
                            # Use softmax sampling if condition is met
                            sampling_accuracy = softmax_sampling_accuracy(
                                logits=predictions['from_squares'][i, 0, :],  # For current sample
                                targets=batch["from_squares"][i].squeeze(0),  # For current sample
                                other_logits=predictions['to_squares'][i, 0, :],  # For current sample
                                other_targets=batch["to_squares"][i].squeeze(0),  # For current sample
                                num_samples=1,
                            )
                        else:
                            # Otherwise, use top-k sampling
                            sampling_accuracy = topk_accuracy_single_sample(
                                logits=predictions['from_squares'][i, 0, :],  # For current sample
                                targets=batch["from_squares"][i].squeeze(0),  # For current sample
                                other_logits=predictions['to_squares'][i, 0, :],  # For current sample
                                other_targets=batch["to_squares"][i].squeeze(0),  # For current sample
                                k=[1, 3, 5],
                            )

                        # Append the sampling accuracy for this sample
                        sampling_accuracy_list.append(sampling_accuracy)

                    # Optionally, you can aggregate the results if needed
                    softmaxsampling_accuracy = torch.mean(torch.tensor(sampling_accuracy_list))


                            
                            
                top1_accuracies.update(top1_accuracy, batch["lengths"].shape[0])
                top3_accuracies.update(top3_accuracy, batch["lengths"].shape[0])
                top5_accuracies.update(top5_accuracy, batch["lengths"].shape[0])
                softmaxsampling_accuracies.update(softmaxsampling_accuracy, batch["lengths"].shape[0])
                
                if rank==0:
                    pbar.update(1)
                if rank==0 and pbar.n>=total_steps:
                    print("yes")
                    pbar.close()
                    break
                            
                            
            if rank==0:
                
                print(f"using range: {range_values[s]}")
                print(f"Move prediction accuracy: {softmaxsampling_accuracies.avg}")
                print("Validation loss: %.3f" % losses.avg)
                print("Validation move loss: %.3f" % move_losses.avg)
                print("Validation result loss: %.3f" % result_losses.avg)
                print("Validation move time loss: %.3f" % move_time_losses.avg)
                print("Validation moves until end loss: %.3f" % moves_until_end_losses.avg)
                print("Validation Categorical game result loss: %.3f" % categorical_game_result_losses.avg)
                print("Validation Categorical game result accuracy: %.3f" % categorical_game_result_accuracies.avg)
                print("Validation top-1 accuracy: %.3f" % top1_accuracies.avg)
                print("Validation top-3 accuracy: %.3f" % top3_accuracies.avg)
                print("Validation top-5 accuracy: %.3f" % top5_accuracies.avg)
                
                s+=1
                print("\n\n")
                pbar = tqdm(total=total_steps, desc="Validating")
                softmaxsampling_accuracies.reset()
                losses.reset()
                result_losses.reset()
                move_time_losses.reset()
                moves_until_end_losses.reset()
                categorical_game_result_accuracies.reset()
                categorical_game_result_losses.reset()
                top1_accuracies.reset()
                top3_accuracies.reset()
                top5_accuracies.reset()
                move_losses.reset()
                """pbar.close()
                cleanup_ddp()
                sys.exit()"""
                        
                                
                
                """top1_accuracies.update(top1_accuracy, batch["lengths"].shape[0])
                top3_accuracies.update(top3_accuracy, batch["lengths"].shape[0])
                top5_accuracies.update(top5_accuracy, batch["lengths"].shape[0])"""
                
                """if predictions['categorical_game_result'] is None:
                    categorical_game_result_accuracies.update(0.0, batch['lengths'].shape[0])
                else:
                    categorical_game_result_accuracies.update(calculate_accuracy(predictions['categorical_game_result'].float(),
                                batch['categorical_result']), batch["lengths"].shape[0])
                if rank==0:
                    pbar.update(1)
                
                total_batches_processed += 1
                if i+1>=total_steps and rank==0:
                    print("validation finished")
                    pbar.close()
                    cleanup_ddp()
                    break
            
            datapoints_skipped = (len(val_loader)-total_batches_processed)*CONFIG.BATCH_SIZE"""
                
            
            
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