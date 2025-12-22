"""
Train a RegressLM on a custom dataset.
Supports JSON formats.
"""

import argparse
import json
import os
import pathlib
import random
from typing import List
from tqdm import tqdm

import numpy as np
import torch
from torch.optim import lr_scheduler
from torch import optim

from regress_lm import core
from regress_lm import vocabs
from regress_lm import tokenizers
from regress_lm.pytorch import model as model_lib
from regress_lm.pytorch import training
from regress_lm.pytorch import data_utils


def preprocess_ghs_example(item, ghs_map, add_other_features: bool = False):
    """
    Custom preprocessing for PubChem/GHS task.
    x: ALL features EXCLUDING 'GHS Codes' + 'Dosage'.
    y: 'Hazards' list from 'GHS Codes', joined as string.
    """
    # 1. Extract Hazards
    ghs_codes = item.get('GHS Codes', {})
    hazards = ghs_codes.get('Hazards', [])
    if not isinstance(hazards, list):
      raise ValueError(f"Expected 'Hazards' to be a list, but got {type(hazards)}")
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
    else:
        raise ValueError("Missing 'SMILES' key in input data")

    x_obj['Dosage'] = dosages
    
    # Add remaining keys
    if add_other_features:
        for k, v in item.items():
            if k not in ['SMILES', 'GHS Codes', 'GHS Classification', 'Hazards Summary', 'Health Hazards']:
                x_obj[k] = v
    
    x_str = json.dumps(x_obj)
    
    # 4. Construct Y
    # Return list of hazard codes
    # If using HazardCodeTokenizer, y should be List[str]
    y_val = hazards 
    
    return x_str, y_val


def load_data(path: str, ghs_map: dict[str, str]) -> List[core.Example]:
    """Loads data from JSON file."""
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
            for item in tqdm(data):
                try:
                    x_val, y_val = preprocess_ghs_example(item, ghs_map)
                    examples.append(core.Example(x=x_val, y=y_val)) 
                except Exception as e:
                    pass
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .json")
        
    print(f"Loaded {len(examples)} examples from {path}")
    return examples


def evaluate_model(model: model_lib.PyTorchModel, examples: List[core.Example], batch_size=16):
    model.eval()
    print("Evaluating...")
    
    total_precision = 0.0
    total_recall = 0.0
    count = 0
    
    ds = data_utils.ExampleDataset(examples)
    dl = torch.utils.data.DataLoader(
        ds, 
        batch_size=batch_size, 
        collate_fn=model.converter.convert_examples,
        shuffle=False
    )
    
    def compute_instance_metrics(args: tuple[core.Example, np.ndarray]):
        ex, pred_obj_row = args
        pred_set = {p for p in pred_obj_row if isinstance(p, str) and p.startswith('H')}
        gold_set = set(ex.y)
        
        tp = len(gold_set.intersection(pred_set))
        prec = tp / len(pred_set) if len(pred_set) > 0 else 0.0
        rec = tp / len(gold_set) if len(gold_set) > 0 else 0.0
        return prec, rec
             
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dl)):
            # batch is dict. decode returns (ids, output_objs)
            # output_objs: (B, num_samples, max_num_objs)
            _, output_objs = model.decode(batch, num_samples=1)
            
            start_idx = i * batch_size
            current_batch_size = output_objs.shape[0]
            batch_examples = examples[start_idx : start_idx + current_batch_size]

            # output_objs[:, 0] gives the first sample for each item in batch
            batch_results = list(map(compute_instance_metrics, zip(batch_examples, output_objs[:, 0])))
            
            batch_prec, batch_rec = zip(*batch_results)
            total_precision += sum(batch_prec)
            total_recall += sum(batch_rec)
            count += current_batch_size
                 
    avg_prec = total_precision / count if count > 0 else 0.0
    avg_rec = total_recall / count if count > 0 else 0.0
    
    print(f"Evaluation Results on {count} instances:")
    print(f"  Average Precision: {avg_prec:.4f}")
    print(f"  Average Recall:    {avg_rec:.4f}")


def train_vocab(examples: List[core.Example], vocab_size: int, output_prefix: str, include_targets: bool = False) -> vocabs.SentencePieceVocab:
    """Trains a SentencePiece vocabulary from inputs (and optionally targets)."""
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


def main():
    parser = argparse.ArgumentParser(description='Train RegressLM on custom data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data')
    parser.add_argument('--ghs_path', type=str, default=None, help='Path to ghs_hazard_statements.json for custom preprocessing')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--vocab_size', type=int, default=8192, help='Vocabulary size')
    parser.add_argument('--max_input_len', type=int, default=512, help='Max input length')
    parser.add_argument('--max_decode_len', type=int, default=75, help='Max decode length (for text gen)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting')

    args = parser.parse_args()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading GHS map from {args.ghs_path}")
    with open(args.ghs_path, 'r') as f:
        ghs_map = json.load(f)

    # 1. Load Data
    print(f"Loading data from {args.data_path}")
    all_examples = load_data(args.data_path, ghs_map=ghs_map)
    
    # 2. Split Data (80/20)
    print(f"Splitting data with seed {args.seed}...")
    rng = random.Random(args.seed)
    rng.shuffle(all_examples)
    
    split_idx = int(len(all_examples) * 0.8)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    print(f"Train size: {len(train_examples)}")
    print(f"Val size: {len(val_examples)}")

    # 3. Setup Vocabulary
    vocab_prefix = os.path.join(args.output_dir, 'sentencepiece')
    vocab_model_path = vocab_prefix + '.model'
    
    if os.path.exists(vocab_model_path):
        print(f"Loading existing vocabulary from {vocab_model_path}")
        encoder_vocab = vocabs.SentencePieceVocab(vocab_model_path)
    else:
        print("Training new vocabulary...")
        encoder_vocab = train_vocab(train_examples, args.vocab_size, vocab_prefix)

    # Decoder Vocab Selection
    # GHS Mode: Use HazardCodeTokenizer
    print("Using HazardCodeTokenizer for Decoder (GHS Mode)")
    decoder_vocab = vocabs.DecoderVocab(tokenizers.HazardCodeTokenizer())
    # Since we predict a list of objects (hazards), and each hazard is 1 object (1 token),
    # max_num_objs determines how many hazards we can predict.
    max_num_objs = args.max_decode_len 

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
    print("Starting training...")
    steps_per_epoch = len(train_examples) // args.batch_size
    train_dl = trainer.train_dl
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        epoch_losses = []
        for i, batch in enumerate(train_dl):
            metrics = trainer.run_train_step(batch)
            loss = metrics.get('train_loss_mean', 0.0)
            epoch_losses.append(loss)
            
            if i % 10 == 0:
                print(f"  Step {i}/{steps_per_epoch} - Loss: {loss:.4f}")
                
        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        val_metrics = trainer.run_validation_epoch()
        print(f"  Val Loss: {val_metrics.get('validation_loss', -1.0):.4f}")
            
        # Save Checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        trainer.save_checkpoint(checkpoint_path)
        print(f"  Saved checkpoint to {checkpoint_path}")

    # 6. Evaluation
    print("\nRunning final evaluation on validation set...")
    evaluate_model(model, val_examples, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
