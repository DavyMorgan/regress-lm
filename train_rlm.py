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

def get_nested(data, key_path):
    """Accesses nested dictionary keys via dot notation."""
    keys = key_path.split('.')
    val = data
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return None
    return val

def load_ghs_map(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def preprocess_ghs_example(item, ghs_map):
    """
    Custom preprocessing for PubChem/GHS task.
    x: ALL features EXCLUDING 'GHS Codes' + 'Dosage'.
    y: 'Hazards' list from 'GHS Codes', joined as string.
    """
    # 1. Extract Hazards
    ghs_codes = item.get('GHS Codes', {})
    hazards = ghs_codes.get('Hazards', [])
    if not isinstance(hazards, list):
         # If hazards is not a list, try to handle it or skip. 
         # Based on inspection, it can be a dict sometimes? 
         # Checked file: "Hazards": ["H302"...] or dict with Ref/Value?
         # Step 161 showed "Hazards": ["H302"...] inside "GHS Codes" dict {"Hazards": [], "Precaution": []}.
         # But in Line 306: "GHS Codes": { "Hazards": [...], ... }.
         pass

    # 2. Dosage Lookup
    # Dosage is list of Categories for each Hazard code.
    dosages = []
    for h_code in hazards:
        info = ghs_map.get(h_code)
        if info:
            dosages.append(info.get('category', 'Unknown'))
        else:
            dosages.append('Unknown')
    
    # 3. Construct X
    # Reorder keys: SMILES, Dosage, then others
    x_obj = {}
    
    if 'SMILES' in item:
        x_obj['SMILES'] = item['SMILES']
        
    x_obj['Dosage'] = dosages
    
    # Add remaining keys
    for k, v in item.items():
        if k not in ['SMILES', 'GHS Codes']:
            x_obj[k] = v
    
    x_str = json.dumps(x_obj)
    
    # 4. Construct Y as string (Output of decoder)
    y_str = " ".join(hazards)
    
    return x_str, y_str


def load_data(path: str, x_key: str = 'x', y_key: str = 'y', ghs_map = None) -> List[core.Example]:
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
                if ghs_map:
                    # GHS Special Logic
                    try:
                        x_val, y_val = preprocess_ghs_example(item, ghs_map)
                        # NOTE: y is string now. passing to float(y) would fail if we didn't update Example.
                        # core.Example.y is type hinted float|Sequence[float].
                        # We are passing str. This will work if we adjust the model to handle text targets.
                        examples.append(core.Example(x=x_val, y=y_val)) 
                    except Exception as e:
                        # Skip malformed
                        pass
                else:
                    # Standard Logic
                    x_val = get_nested(item, x_key)
                    y_val = get_nested(item, y_key)
                    if x_val is not None and y_val is not None:
                        try:
                            examples.append(core.Example(x=str(x_val), y=float(y_val)))
                        except (ValueError, TypeError):
                            pass 
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .json, .jsonl, or .tsv")
        
    print(f"Loaded {len(examples)} examples from {path}")
    return examples

def train_vocab(examples: List[core.Example], vocab_size: int, output_prefix: str, include_targets: bool = False) -> vocabs.SentencePieceVocab:
    """Trains a SentencePiece vocabulary from example inputs (and optionally targets)."""
    # Write inputs to a temporary file
    temp_corpus = output_prefix + ".corpus.txt"
    with open(temp_corpus, 'w') as f:
        for ex in examples:
            f.write(ex.x + '\n')
            if include_targets and isinstance(ex.y, str):
                 f.write(ex.y + '\n')
            
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

class SPDecoderAdapter:
    """Adapts a SentencePieceVocab to behave like a DecoderVocab for PyTorchModel."""
    def __init__(self, sp_vocab: vocabs.SentencePieceVocab):
        self.sp_vocab = sp_vocab
        
    @property
    def num_tokens_per_obj(self) -> int:
        # Treat each generated token as 1 object unit (for text generation)
        return 1
        
    @property
    def bos_pad_id(self) -> int:
        return self.sp_vocab.pad_id
        
    def to_token_ids(self, obj, /) -> list[int]:
        # obj is expected to be a string (the target text)
        if isinstance(obj, list):
            # If it's a list (e.g. list of strings?), join them or handle each?
            # For this task y is a single string "H302 H314"
            # But core.Example might have passed a list if we didn't join it.
            # We implemented joining in preprocess_ghs_example.
            pass
        return self.sp_vocab.to_token_ids(obj)

    def __len__(self) -> int:
        return len(self.sp_vocab)

def main():
    parser = argparse.ArgumentParser(description='Train RegressLM on custom data')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_path', type=str, default=None, help='Path to validation data')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--vocab_size', type=int, default=8192, help='Vocabulary size')
    parser.add_argument('--max_input_len', type=int, default=512, help='Max input length')
    parser.add_argument('--max_decode_len', type=int, default=128, help='Max decode length (for text gen)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--x_key', type=str, default='x', help='Key for input text (supports dot notation)')
    parser.add_argument('--y_key', type=str, default='y', help='Key for target value (supports dot notation)')
    
    # GHS specific args
    parser.add_argument('--ghs_path', type=str, default=None, help='Path to ghs_hazard_statements.json for custom preprocessing')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load GHS map if provided
    ghs_map = None
    if args.ghs_path:
        print(f"Loading GHS map from {args.ghs_path}")
        ghs_map = load_ghs_map(args.ghs_path)

    # 1. Load Data
    train_examples = load_data(args.train_path, x_key=args.x_key, y_key=args.y_key, ghs_map=ghs_map)
    val_examples = load_data(args.val_path, x_key=args.x_key, y_key=args.y_key, ghs_map=ghs_map) if args.val_path else []

    # 2. Setup Vocabulary
    vocab_prefix = os.path.join(args.output_dir, 'sentencepiece')
    vocab_model_path = vocab_prefix + '.model'
    
    if os.path.exists(vocab_model_path):
        print(f"Loading existing vocabulary from {vocab_model_path}")
        encoder_vocab = vocabs.SentencePieceVocab(vocab_model_path)
    else:
        print("Training new vocabulary...")
        # If GHS mode (text targets), should we include targets in vocab training?
        # Yes, useful for decoding hazard codes tokens properly.
        include_targets = (ghs_map is not None)
        encoder_vocab = train_vocab(train_examples, args.vocab_size, vocab_prefix, include_targets=include_targets)

    # Decoder Vocab Selection
    max_num_objs = 1 
    if ghs_map:
        # Text Generation Mode: Use Same SentencePiece Vocab for Decoder (via adapter)
        print("Using SentencePiece vocabulary for Decoder (Text Generation Mode)")
        decoder_vocab = SPDecoderAdapter(encoder_vocab)
        max_num_objs = args.max_decode_len # Set generated length
    else:
        # Standard Regression Mode: Use P10 Tokenizer
        print("Using P10 Tokenizer for Decoder (Regression Mode)")
        decoder_vocab = vocabs.DecoderVocab(tokenizers.P10Tokenizer())
        max_num_objs = 1 # Regression typically outputs 1 value (or small fixed seq)

    # 3. Model Configuration
    config = model_lib.PyTorchModelConfig(
        encoder_vocab=encoder_vocab,
        decoder_vocab=decoder_vocab,
        max_input_len=args.max_input_len,
        max_num_objs=max_num_objs, # Important for Text Gen
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
