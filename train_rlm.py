"""
Train a RegressLM on a custom dataset.
Supports JSON, JSONL, and TSV formats.
"""

import argparse
import json
import os
import pathlib
import sys
from typing import List, Tuple

import torch
from torch.optim import lr_scheduler
from torch import optim

from regress_lm import core
from regress_lm import vocabs
from regress_lm import tokenizers
from regress_lm.pytorch import model as model_lib
from regress_lm.pytorch import training
from regress_lm.pytorch import data_utils

def load_data(path: str) -> List[core.Example]:
    """Loads data from JSON, JSONL, or TSV file."""
    examples = []
    path = pathlib.Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    
    if ext == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of objects.")
            for item in data:
                examples.append(core.Example(x=str(item['x']), y=float(item['y'])))
                
    elif ext == '.jsonl':
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                examples.append(core.Example(x=str(item['x']), y=float(item['y'])))
                
    elif ext == '.tsv':
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    examples.append(core.Example(x=parts[0], y=float(parts[1])))
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .json, .jsonl, or .tsv")
        
    print(f"Loaded {len(examples)} examples from {path}")
    return examples

def train_vocab(examples: List[core.Example], vocab_size: int, output_prefix: str) -> vocabs.SentencePieceVocab:
    """Trains a SentencePiece vocabulary from example inputs."""
    # Write inputs to a temporary file
    temp_corpus = output_prefix + ".corpus.txt"
    with open(temp_corpus, 'w') as f:
        for ex in examples:
            f.write(ex.x + '\n')
            
    print(f"Training vocabulary of size {vocab_size} on {temp_corpus}...")
    vocab = vocabs.SentencePieceVocab.from_corpus(
        corpus_path=temp_corpus,
        vocab_size=vocab_size,
        model_prefix=output_prefix
    )
    
    # Clean up temp file
    if os.path.exists(temp_corpus):
        os.remove(temp_corpus)
        
    return vocab

def main():
    parser = argparse.ArgumentParser(description='Train RegressLM on custom data')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_path', type=str, default=None, help='Path to validation data')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--vocab_size', type=int, default=8192, help='Vocabulary size')
    parser.add_argument('--max_input_len', type=int, default=512, help='Max input length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    train_examples = load_data(args.train_path)
    val_examples = load_data(args.val_path) if args.val_path else []

    # 2. Setup Vocabulary
    vocab_prefix = os.path.join(args.output_dir, 'sentencepiece')
    vocab_model_path = vocab_prefix + '.model'
    
    if os.path.exists(vocab_model_path):
        print(f"Loading existing vocabulary from {vocab_model_path}")
        encoder_vocab = vocabs.SentencePieceVocab(vocab_model_path)
    else:
        print("Training new vocabulary...")
        encoder_vocab = train_vocab(train_examples, args.vocab_size, vocab_prefix)

    # Standard P10 tokenizer for regression targets
    decoder_vocab = vocabs.DecoderVocab(tokenizers.P10Tokenizer())

    # 3. Model Configuration
    config = model_lib.PyTorchModelConfig(
        encoder_vocab=encoder_vocab,
        decoder_vocab=decoder_vocab,
        max_input_len=args.max_input_len,
        architecture_kwargs={
            'd_model': 512, 
            'num_encoder_layers': 6, 
            'num_decoder_layers': 6,
        }
    )
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = config.make_model(compile_model=False) 
    model.to(device)

    # 4. Training Setup
    optimizer_factory = lambda params: optim.AdamW(params, lr=args.lr)
    scheduler_factory = lambda opt: lr_scheduler.StepLR(opt, step_size=1, gamma=0.95) # Simple decay

    train_ds = data_utils.ExampleDataset(train_examples)
    val_ds = data_utils.ExampleDataset(val_examples) if val_examples else data_utils.ExampleDataset(train_examples[:10]) # Use subset of train if no val

    trainer = training.Trainer(
        model=model,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        train_ds=train_ds,
        validation_ds=val_ds,
        batch_size=args.batch_size,
    )

    # 5. Training Loop
    print("Starting training...")
    steps_per_epoch = len(train_examples) // args.batch_size
    
    # Simply using the cycle_dataloader/run_train_step pattern manually or iterating
    # Since the Trainer doesn't hold the loop itself, we write it here.
    
    train_dl = trainer.train_dl
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        epoch_losses = []
        for i, batch in enumerate(train_dl):
            # Move batch to device is handled inside model.compute_losses_and_metrics 
            # via to_device, but Trainer usually handles it? 
            # Looking at training.py, Trainer.run_train_step calls wrapper.forward(batch)
            # and wrapper calls model.compute_losses_and_metrics(batch).
            # model.compute_losses_and_metrics calls self.to_device(examples['...'])
            # So we just pass the batch from dataloader.
            
            # Note: The trainer implementation in training.py has a slightly complex run_train_step 
            # which updates weights. 
            
            metrics = trainer.run_train_step(batch)
            loss = metrics.get('train_loss_mean', 0.0)
            epoch_losses.append(loss)
            
            if i % 10 == 0:
                print(f"  Step {i}/{steps_per_epoch} - Loss: {loss:.4f}")
                
        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        if val_examples:
            val_metrics = trainer.run_validation_epoch()
            print(f"  Val Loss: {val_metrics.get('validation_loss', -1.0):.4f}")
            
        # Save Checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        trainer.save_checkpoint(checkpoint_path)
        print(f"  Saved checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    main()
