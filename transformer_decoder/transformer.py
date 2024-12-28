import tensorflow as tf
import math
import argparse
import yaml
import time
import numpy as np
import sys
import os
from loss import LabelSmoothedCE

from modules import BoardEncoder, MoveDecoder

#from pgn_parser import dataset, val_dataset, warmup_dataset, batch_size
#from pgn_parser import val_dataset, warmup_dataset, batch_size
#from dataloading_test import dataset
from datasets import ChessDataset

from lr_schedule import get_lr
from utils import topk_accuracy2, AverageMeter, change_lr

from pathlib import Path

log_dir = 'logs/fit/'

# Ensure the log directory exists
Path(log_dir).mkdir(parents=True, exist_ok=True)

# Get a list of all subdirectories in the log directory
subdirs = [int(d.name) for d in Path(log_dir).iterdir() if d.is_dir() and d.name.isdigit()]

# Determine the next log directory number
next_log_number = max(subdirs, default=0) + 1

# Create the new log directory
new_log_dir = Path(log_dir) / str(next_log_number)
new_log_dir.mkdir()

print(f'Created new log directory: {new_log_dir}')
log_dir = new_log_dir

# Create a Summary Writer
writer = tf.summary.create_file_writer(str(log_dir))

class ChessTransformer(tf.keras.Model):
    def __init__(self, cfg):
        super(ChessTransformer, self).__init__()

        self.code = "ED"

        self.vocab_sizes = cfg['model']['vocab_sizes']
        self.n_moves = cfg['model']['n_moves']
        self.d_model = cfg['model']['d_model']
        self.n_heads = cfg['model']['n_heads']
        self.d_queries = cfg['model']['d_queries']
        self.d_values = cfg['model']['d_values']
        self.d_inner = cfg['model']['d_inner']
        self.n_layers = cfg['model']['n_layers']
        self.dropout = cfg['model']['dropout']

        # Encoder
        self.board_encoder = BoardEncoder(
            vocab_sizes=self.vocab_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        # Decoder
        self.move_decoder = MoveDecoder(
            vocab_size=self.vocab_sizes["moves"],
            n_moves=self.n_moves,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights in the transformer model.
        """
        # Glorot uniform initialization
        for layer in self.layers:
            for weight in layer.weights:
                if len(weight.shape) > 1:  # Check for at least two dimensions
                    initializer = tf.keras.initializers.GlorotUniform()
                    weight.assign(initializer(weight.shape))
    def call(self, inputs):
        # Encoder
        boards = self.board_encoder(
            inputs["turns"],
            inputs["white_kingside_castling_rights"],
            inputs["white_queenside_castling_rights"],
            inputs["black_kingside_castling_rights"],
            inputs["black_queenside_castling_rights"],
            inputs["board_positions"],
        )  # (N, BOARD_STATUS_LENGTH, d_model)

        # Decoder
        moves = self.move_decoder(
            inputs["moves"][:, :-1], inputs["lengths"], boards
        )  # (N, n_moves, move_vocab_size)
        # Note: We don't pass the last move as it has no next-move

        return moves
    

    
with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
model = ChessTransformer(cfg)
"""for i, batch in enumerate(warmup_dataset):
    model(batch[0])
    break
model.load_weights('model_weights/model_weights_step_22500.weights.h5')"""


criterion = LabelSmoothedCE(eps=cfg['training']['label_smoothing'],
                       n_predictions=cfg['training']['n_moves'])

LR = get_lr(
    step=cfg['training']['step'],
    d_model=cfg['model']['d_model'],
    warmup_steps=cfg['training']['warmup_steps'],
    schedule=cfg['training']['lr_schedule'],
    decay=cfg['training']['lr_decay'],
) 
optimizer = tf.keras.optimizers.Adam(
    learning_rate=LR,
    beta_1=cfg['training']['betas'][0],  
    beta_2=cfg['training']['betas'][1],  
    epsilon=float(cfg['training']['epsilon'])
)
#optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer)


# Track some metrics
data_time = AverageMeter()  # data loading time
step_time = AverageMeter()  # forward prop. + back prop. time
losses = AverageMeter()  # loss
top1_accuracies = AverageMeter()  # top-1 accuracy of first move
top3_accuracies = AverageMeter()  # top-3 accuracy of first move
top5_accuracies = AverageMeter()  # top-5 accuracy of first move

val_losses = AverageMeter()
top1_accuracies_val = AverageMeter()  # top-1 validation accuracy of first move
top3_accuracies_val = AverageMeter()  # top-3 validation accuracy of first move
top5_accuracies_val = AverageMeter()  # top-5 validation accuracy of first move


start_step_time = time.time()

print("start of epoch 1")
step = cfg['training']['step']
epoch = 1 #current epoch
epochs = 3 #total epochs
dataset_len = 40000

accumulated_gradients = []

#generate array for accumulating gradients
dataset = ChessDataset(
    #data_folder="data/epoch_1/train_data/",
    data_folder='../../../drive/My Drive/data/epoch_1/train_data/',
    h5_file="data.h5",
    n_moves=10
).as_tensorflow_dataset().batch(1)
model_copy = model
for i, batch in enumerate(dataset):
    model_copy(batch)
    break
model_copy.board_encoder.summary()
model_copy.move_decoder.summary()
accumulated_gradients = [tf.zeros_like(var) for var in model_copy.trainable_weights]


for epoch in range(epochs):
    epoch = epoch + 1
    
    train_dataset = ChessDataset(
        #data_folder=f"data/epoch_{epoch}/train_data/",
        data_folder=f'../../../drive/My Drive/data/epoch_{epoch}/train_data/',
        h5_file="data.h5",
        n_moves=10
    )
    val_dataset = ChessDataset(
        #data_folder=f"data/epoch_{epoch}/val_data/",
        data_folder=f'../../../drive/My Drive/data/epoch_{epoch}/val_data/',
        h5_file="data.h5",
        n_moves=10
    )
    
    train_dataset_length = train_dataset.length
    val_dataset_length = val_dataset.length
    batch_size = cfg['dataloading']['batch_size']
    train_dataset = train_dataset.as_tensorflow_dataset().shuffle(buffer_size=int(float(cfg['dataloading']['buffer_size']))).prefetch(tf.data.AUTOTUNE).batch(batch_size).cache()
    val_dataset = val_dataset.as_tensorflow_dataset().shuffle(val_dataset.length).batch(val_dataset.length).take(1).cache()
    
    
    i=step*4
    start_data_time = time.time()

    for batch in train_dataset:
        # Time taken to load data
        #print(time.time() - start_data_time)
        data_time.update(time.time() - start_data_time)
        start_data_time = time.time()
        
        i+=1
        
        with tf.GradientTape() as tape:
            predicted_moves = model(batch)  # (N, n_moves, move_vocab_size)

            with tf.device('/CPU:0'):
                loss = criterion(
                    y_true=batch,  # batch["encoded_moves"][:, 1:]  # (N, n_moves)
                    y_pred=predicted_moves,  # (N, n_moves, move_vocab_size)
                    # lengths=batch["lengths"],  # (N, 1)
                )
            
            loss = loss / cfg['training']['batches_per_step'] 
            
        gradients = tape.gradient(loss, model.trainable_weights)
            
        lengths_sum = tf.reduce_sum(batch["lengths"])  # Sum the tensor
        lengths_sum_value = float(lengths_sum.numpy())
        losses.update(
            loss.numpy() * cfg['training']['batches_per_step'], lengths_sum_value
        )

        # Compute accuracies on CPU to save GPU memory
        with tf.device('/CPU:0'):
            top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy2(
                logits=predicted_moves[:, 0, :],
                targets=batch["moves"][:, 1],
                k=[1, 3, 5],
            )
        
        
        for j, grad in enumerate(gradients):
            if isinstance(grad, tf.IndexedSlices):
                grad = tf.convert_to_tensor(grad)
            accumulated_gradients[j] += grad #/ cfg['training']['batches_per_step']
            
        
        #print(top1_accuracy)
        
        
        top1_accuracies.update(top1_accuracy, batch["lengths"].shape[0])
        top3_accuracies.update(top3_accuracy, batch["lengths"].shape[0])
        top5_accuracies.update(top5_accuracy, batch["lengths"].shape[0])
        
        
        # Update model (i.e. perform a training step) only after
        # gradients are accumulated from batches_per_step batches
        if (i + 1) % cfg['training']['batches_per_step'] == 0:
            optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_weights))
            accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_weights]

            # This step is now complete
            step += 1

            if step % 250 == 0:
                print(f"{step} steps reached")

                # Define the directory for saving model weights
                model_weights_dir = "model_weights"

                # Create the directory if it doesn't exist
                if not os.path.exists(model_weights_dir):
                    os.makedirs(model_weights_dir)
                    print(f"Created directory: {model_weights_dir}")

                # Check how many files are in the directory
                files = os.listdir(model_weights_dir)
                if len(files) > 5:
                    # Delete all files in the directory
                    for file in files:
                        os.remove(os.path.join(model_weights_dir, file))
                    print("Deleted all files in the directory")

                # Save model weights
                model.save_weights(f"{model_weights_dir}/model_weights_step_{step}.weights.h5")
                print("Weights saved")

                
            new_lr = get_lr(
                        step=step,
                        d_model=cfg['model']['d_model'],
                        warmup_steps=cfg['training']['warmup_steps'],
                        schedule=cfg['training']['lr_schedule'],
                        decay=cfg['training']['lr_decay'],
                    )
            
            change_lr(optimizer,
                    new_lr=get_lr(
                        step=step,
                        d_model=cfg['model']['d_model'],
                        warmup_steps=cfg['training']['warmup_steps'],
                        schedule=cfg['training']['lr_schedule'],
                        decay=cfg['training']['lr_decay'],
                    ),
                )
            
            if (i+1) % cfg['training']['steps_per_epoch'] == 0:
                start_val_time = time.time()
                for batch in val_dataset:
                    batch = batch
                    predicted_moves = model(batch, training=False)  # (N, n_moves, move_vocab_size)

                    # Loss
                    with tf.device('/CPU:0'):
                        loss = criterion(
                            y_true=batch,  # batch["encoded_moves"][:, 1:]  # (N, n_moves)
                            y_pred=predicted_moves,  # (N, n_moves, move_vocab_size)
                            # lengths=batch["lengths"],  # (N, 1)
                        )
                    lengths_sum = tf.reduce_sum(batch["lengths"])  # Sum the tensor
                    lengths_sum_value = float(lengths_sum.numpy())
                    val_losses.update(
                        loss.numpy(), lengths_sum_value
                    )
                    # Compute accuracies on CPU to save GPU memory
                    with tf.device('/CPU:0'):
                        top1_accuracy_val, top3_accuracy_val, top5_accuracy_val = topk_accuracy2(
                            logits=predicted_moves[:, 0, :],
                            targets=batch["moves"][:, 1],
                            k=[1, 3, 5],
                        )
                    top1_accuracies_val.update(top1_accuracy_val, batch["lengths"].shape[0])
                    top3_accuracies_val.update(top3_accuracy_val, batch["lengths"].shape[0])
                    top5_accuracies_val.update(top5_accuracy_val, batch["lengths"].shape[0])
                elapsed_val_time = time.time() - start_val_time
                print(f"validation time: {elapsed_val_time:.4f}s")


            # Time taken for this training step
            step_time.update(time.time() - start_step_time)
            start_step_time = time.time()

            # Print status
            if step % cfg['training']['print_frequency'] == 0:
                if (i+1) % cfg['training']['steps_per_epoch'] == 0:
                    print(
                        "Epoch {0}/{1}---"
                        "Batch {2}/{3}---"
                        "Step {4}/{5}---"
                        "Data Time {data_time.val:.3f} ({data_time.avg:.3f})---"
                        "Step Time {step_time.val:.3f} ({step_time.avg:.3f})---"
                        "Loss {losses.val:.4f} ({losses.avg:.4f})---"
                        "Top-1 {top1s.val:.4f} ({top1s.avg:.4f})"
                        "Top-3 {top3s.val:.4f} ({top3s.avg:.4f})"
                        "Top-5 {top5s.val:.4f} ({top5s.avg:.4f})"
                        "Validation Loss {val_losses.val:.4f} ({val_losses.avg:.4f})---"
                        "Validation Top-1 {val_top1s.val:.4f} ({val_top1s.avg:.4f})"
                        "Validation Top-3 {val_top3s.val:.4f} ({val_top3s.avg:.4f})"
                        "Validation Top-5 {val_top5s.val:.4f} ({val_top5s.avg:.4f})".format(
                            epoch,
                            epochs,
                            i + 1,
                            train_dataset_length,
                            step,
                            train_dataset_length//cfg['training']['print_frequency'],
                            step_time=step_time,
                            data_time=data_time,
                            losses=losses,
                            top1s=top1_accuracies,
                            top3s=top3_accuracies,
                            top5s=top5_accuracies,
                            val_losses=val_losses,
                            val_top1s=top1_accuracies_val,
                            val_top3s=top3_accuracies_val,
                            val_top5s=top5_accuracies_val,
                        )
                    )
                else:
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
                            epoch,
                            epochs,
                            i + 1,
                            train_dataset_length,
                            step,
                            train_dataset_length//cfg['training']['print_frequency'],
                            step_time=step_time,
                            data_time=data_time,
                            losses=losses,
                            top1s=top1_accuracies,
                            top3s=top3_accuracies,
                            top5s=top5_accuracies,
                        )
                    )
            with writer.as_default():
                # Log the training loss
                tf.summary.scalar("train/loss", losses.val, step=step)
                
                # Log the learning rate
                tf.summary.scalar("train/lr", new_lr, step=step)
                
                # Log data time
                tf.summary.scalar("train/data_time", data_time.val, step=step)
                
                # Log step time
                tf.summary.scalar("train/step_time", step_time.val, step=step)
                
                # Log top-1 accuracy
                tf.summary.scalar("train/top1_accuracy", top1_accuracies.val, step=step)
                
                # Log top-3 accuracy
                tf.summary.scalar("train/top3_accuracy", top3_accuracies.val, step=step)
                
                # Log top-5 accuracy
                tf.summary.scalar("train/top5_accuracy", top5_accuracies.val, step=step)
                
                #validation metrics
                tf.summary.scalar("validation/loss", val_losses.val, step=step)
                
                # Log top-1 accuracy
                tf.summary.scalar("validation/top1_accuracy", top1_accuracies_val.val, step=step)
                
                # Log top-3 accuracy
                tf.summary.scalar("validation/top3_accuracy", top3_accuracies_val.val, step=step)
                
                # Log top-5 accuracy
                tf.summary.scalar("validation/top5_accuracy", top5_accuracies_val.val, step=step)
                
        #writer.flush()
        #writer.close()

    
    
    
    
    
    """model.fit(dataset,
          epochs=100,
          steps_per_epoch=1000,
          validation_data=val_dataset,
          loss=LabelSmoothedCE
)"""