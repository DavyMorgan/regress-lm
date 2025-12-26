"""
Train a RegressLM on a custom dataset.
Supports JSON formats.
"""

import argparse
import json
import logging
import os
import random
from typing import List
from tqdm import tqdm

import numpy as np
import torch
from torch.optim import lr_scheduler
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from regress_lm import core
from regress_lm import tokenizers
from regress_lm import vocabs
from regress_lm.pytorch import data_utils
from regress_lm.pytorch import encoders
from regress_lm.pytorch import model as model_lib
from regress_lm.pytorch import training

from utils import load_data, evaluate_model


def train_vocab(examples: List[core.Example], vocab_size: int, output_prefix: str, include_targets: bool = False) -> vocabs.SentencePieceVocab:
    """Trains a SentencePiece vocabulary from inputs (and optionally targets)."""
    # Write inputs to a temporary file
    temp_corpus = output_prefix + ".corpus.txt"
    with open(temp_corpus, 'w') as f:
        for ex in examples:
            f.write(ex.x + '\n')
            if include_targets and isinstance(ex.y, str):
                 f.write(ex.y + '\n')
            
    logging.info(f"Training vocabulary of size {vocab_size} on {temp_corpus}...")
    vocab = vocabs.SentencePieceVocab.from_corpus(
        corpus_path=temp_corpus,
        vocab_size=vocab_size,
        model_prefix=output_prefix
    )
    
    # Clean up temp file
    if os.path.exists(temp_corpus):
        os.remove(temp_corpus)
        
    return vocab


def get_scheduler(optimizer, warmup_steps, hold_steps, total_steps, init_lr, min_lr):
    """
    Creates a learning rate scheduler with warmup, hold, and cosine decay phases.
    """
    min_multiplicative_factor = min_lr / init_lr
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < warmup_steps + hold_steps:
            return 1.0
        else:
            # Cosine decay
            decay_steps = total_steps - warmup_steps - hold_steps
            if decay_steps <= 0:
                return min_multiplicative_factor
            
            progress = float(current_step - warmup_steps - hold_steps) / float(max(1, decay_steps))
            return max(min_multiplicative_factor, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description='Train RegressLM on custom data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data')
    parser.add_argument('--ghs_path', type=str, default='ghs_hazard_statements.json', help='Path to ghs_hazard_statements.json for custom preprocessing')
    parser.add_argument('--keys_path', type=str, default='keys_classification.json', help='Path to keys_classification.json for custom preprocessing')
    parser.add_argument('--add_auxiliary_features', action='store_true', help='Add auxiliary features to input')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--vocab_size', type=int, default=8192, help='Vocabulary size')
    parser.add_argument('--max_input_len', type=int, default=512, help='Max input length')
    parser.add_argument('--max_num_hazard_codes', type=int, default=30, help='Max number of hazard codes per example')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Number of warmup steps')
    parser.add_argument('--hold_steps', type=int, default=1000, help='Number of hold steps')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum LR after decay')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting')
    parser.add_argument('--encoder_type', type=str, default='vanilla', choices=['vanilla', 't5gemma'], help='Encoder type')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of generated responses per example')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for decoding')

    args = parser.parse_args()

    if args.num_samples > 1:
        assert args.temperature > 0.0, "Temperature must be > 0.0 for num_samples > 1"
    
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=args.output_dir)
    
    # 1. Load Data
    logging.info(f"Loading GHS map from {args.ghs_path}")
    with open(args.ghs_path, 'r') as f:
        ghs_map = json.load(f)

    logging.info(f"Loading keys classification from {args.keys_path}")
    with open(args.keys_path, 'r') as f:
        keys_map = json.load(f)

    logging.info(f"Loading data from {args.data_path}")
    all_examples = load_data(args.data_path, ghs_map=ghs_map, keys_map=keys_map, add_auxiliary_features=args.add_auxiliary_features)
    
    # 2. Split Data (80/20)
    logging.info(f"Splitting data with seed {args.seed}...")
    rng = random.Random(args.seed)
    rng.shuffle(all_examples)
    
    split_idx = int(len(all_examples) * 0.8)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    logging.info(f"Train size: {len(train_examples)}")
    logging.info(f"Val size: {len(val_examples)}")

    # 3. Setup Vocabulary
    vocab_prefix = os.path.join(args.output_dir, 'sentencepiece')
    vocab_model_path = vocab_prefix + '.model'
    
    if os.path.exists(vocab_model_path):
        logging.info(f"Loading existing vocabulary from {vocab_model_path}")
        encoder_vocab = vocabs.SentencePieceVocab(vocab_model_path)
    else:
        logging.info("Training new vocabulary...")
        encoder_vocab = train_vocab(train_examples, args.vocab_size, vocab_prefix)

    # Decoder Vocab Selection
    # GHS Mode: Use HazardCodeTokenizer
    logging.info("Using HazardCodeTokenizer for Decoder (GHS Mode)")
    decoder_vocab = vocabs.DecoderVocab(
        tokenizers.HazardCodeTokenizer(
            all_hazard_codes=list(ghs_map.keys())+['NULL', 'STOP'],
            max_num_hazard_codes=args.max_num_hazard_codes,
        )
    )

    # 3. Model Configuration
    architecture_kwargs = {
        'd_model': args.d_model,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
    }
    if args.encoder_type == 'vanilla':
        architecture_kwargs['encoder_type'] = encoders.EncoderType.VANILLA
    elif args.encoder_type == 't5gemma':
        architecture_kwargs['encoder_type'] = encoders.EncoderType.T5GEMMA
        architecture_kwargs['additional_encoder_kwargs'] = {'model_name': 'google/t5gemma-s-s-prefixlm'}
    else:
        raise ValueError(f"Unsupported encoder type: {args.encoder_type}")
    config = model_lib.PyTorchModelConfig(
        encoder_vocab=encoder_vocab,
        decoder_vocab=decoder_vocab,
        max_input_len=args.max_input_len,
        max_num_objs=1,
        architecture_kwargs=architecture_kwargs
    )
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    model = config.make_model(compile_model=False) 
    model.to(device)

    # 4. Training Setup
    steps_per_epoch = len(train_examples) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    
    optimizer_factory = lambda params: optim.AdamW(params, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    scheduler_factory = lambda opt: get_scheduler(
        opt, 
        warmup_steps=args.warmup_steps, 
        hold_steps=args.hold_steps, 
        total_steps=total_steps,
        init_lr=args.lr,
        min_lr=args.min_lr
    )

    train_ds = data_utils.ExampleDataset(train_examples)
    val_ds = data_utils.ExampleDataset(val_examples)

    trainer = training.Trainer(
        model=model,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        train_ds=train_ds,
        validation_ds=val_ds,
        batch_size=args.batch_size,
    )

    # 5. Training Loop
    logging.info("Starting training...")
    train_dl = trainer.train_dl
    
    global_step = 0

    logging.info("Running initial evaluation on validation set...")
    _, _, f1 = evaluate_model(model, val_examples, batch_size=args.batch_size, num_samples=1, temperature=0.0, writer=writer, step=global_step)
    best_f1 = f1
    best_step = global_step
    checkpoint_path = os.path.join(args.output_dir, f"best_checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)

    for epoch in tqdm(range(args.epochs), desc="Epoch", leave=True, position=0):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        epoch_losses = []
        for i, batch in enumerate(tqdm(train_dl, desc="Train Step", leave=False, position=1)):
            global_step += 1
            metrics = trainer.run_train_step(batch)
            loss = metrics.get('train_loss_mean', 0.0)
            perplexity = metrics.get('train_perplexity', 0.0)
            epoch_losses.append(loss)
            
            if (i + 1) % (steps_per_epoch // 10) == 0:
                logging.info(f"  Step {i+1}/{steps_per_epoch} - Loss: {loss:.4f}")
                writer.add_scalar('Train/Loss', loss, global_step)
                writer.add_scalar('Train/Perplexity', perplexity, global_step)
                
        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        logging.info(f"  Avg Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar('Train/Avg_Loss', avg_train_loss, global_step)
        
        # Validation
        val_metrics = trainer.run_validation_epoch()
        val_loss = val_metrics.get('validation_loss', -1.0)
        val_ppl = val_metrics.get('validation_perplexity', -1.0)
        logging.info(f"  Val Loss: {val_loss:.4f}")
        
        writer.add_scalar('Val/Loss', val_loss, global_step)
        writer.add_scalar('Val/Perplexity', val_ppl, global_step)

        _, _, f1 = evaluate_model(model, val_examples, batch_size=args.batch_size, num_samples=1, temperature=0.0, writer=writer, step=global_step)
            
        # Save Checkpoint
        if f1 > best_f1:
            best_f1 = f1
            best_step = global_step
            checkpoint_path = os.path.join(args.output_dir, f"best_checkpoint.pt")
            trainer.save_checkpoint(checkpoint_path)
            logging.info(f"  Saved best checkpoint to {checkpoint_path} at epoch {epoch+1}")

    # 6. Evaluation
    logging.info("Running final evaluation of best checkpoint on validation set...")
    checkpoint_path = os.path.join(args.output_dir, f"best_checkpoint.pt")
    trainer.load_checkpoint(checkpoint_path)
    evaluate_model(model, val_examples, batch_size=args.batch_size, num_samples=1, temperature=0.0, writer=None, step=best_step)

    if args.num_samples > 1:
        logging.info("Running best-of-n voting evaluation on validation set...")
        evaluate_model(model, val_examples, batch_size=args.batch_size, num_samples=args.num_samples, temperature=args.temperature, writer=None, step=best_step)
    
    writer.close()
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
