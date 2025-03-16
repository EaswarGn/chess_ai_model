from huggingface_hub import hf_hub_download
import os
import torch
from configs import import_config
import argparse

def average_checkpoints(min_checkpoint_step, max_checkpoint_step, checkpoint_folder, repo_id="codingmonster1234/full_trained_model", output_path="averaged_checkpoint.pt"):
    num_checkpoints = max_checkpoint_step - min_checkpoint_step + 1
    
    avg_state_dict = None
    i = 0
    
    for step in range(min_checkpoint_step, max_checkpoint_step + 1):
        filename = f"checkpoints/models/1900_step_{step}.pt"
        hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=checkpoint_folder)
        if i == 0:
            avg_state_dict = torch.load(f'{checkpoint_folder}/{filename}', map_location="cpu")['model_state_dict']
        else:
            state_dict = torch.load(f'{checkpoint_folder}/{filename}', map_location="cpu")['model_state_dict']
            for key in avg_state_dict.keys():
                avg_state_dict[key] += state_dict[key] 
        i+=1
    
    for key in avg_state_dict.keys():
        avg_state_dict[key] /= num_checkpoints
        
    torch.save({'model_state_dict': avg_state_dict}, output_path)
    
    print(f"Averaged checkpoint saved to {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average model checkpoints.")
    
    parser.add_argument("--min_checkpoint_step", type=int, required=True, help="Minimum checkpoint step")
    parser.add_argument("--max_checkpoint_step", type=int, required=True, help="Maximum checkpoint step")
    parser.add_argument("--checkpoint_folder", type=str, default="checkpoints", help="Folder containing checkpoints")
    parser.add_argument("--output_path", type=str, default="averaged_checkpoint.pt", help="Path to save the averaged checkpoint")

    args = parser.parse_args()

    os.makedirs(args.checkpoint_folder, exist_ok=True)

    average_checkpoints(
        min_checkpoint_step=args.min_checkpoint_step,
        max_checkpoint_step=args.max_checkpoint_step,
        checkpoint_folder=args.checkpoint_folder,
        output_path=args.output_path
    )