"""
Train a RegressLM on a custom dataset.
Supports JSON formats.
"""

import argparse
import json
import logging
import os
import pathlib
import random
from typing import List
from tqdm import tqdm

import numpy as np
import torch
from torch.optim import lr_scheduler
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from regress_lm import core
from regress_lm import vocabs
from regress_lm import tokenizers
from regress_lm.pytorch import model as model_lib
from regress_lm.pytorch import training
from regress_lm.pytorch import data_utils

from ordered_set import OrderedSet


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
    if len(hazards) == 0:
        hazards = ['NULL']
    hazards = sorted(hazards)
    hazards.append('STOP')
    y_str = [' '.join(hazards)]
    
    return x_str, y_str


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
        
    logging.info(f"Loaded {len(examples)} examples from {path}")
    return examples


def evaluate_model(model: model_lib.PyTorchModel, examples: List[core.Example], batch_size=16, writer=None, step: int = 0):
    model.eval()
    logging.info(f"Evaluating at step {step}...")
    
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
    
    def compute_instance_metrics(args: tuple[core.Example, str]):
        ex, pred = args
        pred_set = OrderedSet([c for c in pred.split() if c != 'STOP'])
        gold_set = OrderedSet(ex.y[0].split()[:-1])

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
            batch_results = list(map(compute_instance_metrics, zip(batch_examples, output_objs[:, 0, 0])))
            
            batch_prec, batch_rec = zip(*batch_results)
            total_precision += sum(batch_prec)
            total_recall += sum(batch_rec)
            count += current_batch_size
                 
    avg_prec = total_precision / count if count > 0 else 0.0
    avg_rec = total_recall / count if count > 0 else 0.0
    f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec) if avg_prec + avg_rec > 0 else 0.0
    
    logging.info(f"Evaluation Results on {count} instances:")
    logging.info(f"  Average Precision: {avg_prec:.4f}")
    logging.info(f"  Average Recall:    {avg_rec:.4f}")
    logging.info(f"  Average F1:        {f1:.4f}")

    if writer:
        writer.add_scalar('Eval/Precision', avg_prec, step)
        writer.add_scalar('Eval/Recall', avg_rec, step)
        writer.add_scalar('Eval/F1', f1, step)

    return avg_prec, avg_rec, f1


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


def main():
    parser = argparse.ArgumentParser(description='Train RegressLM on custom data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data')
    parser.add_argument('--ghs_path', type=str, default=None, help='Path to ghs_hazard_statements.json for custom preprocessing')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--vocab_size', type=int, default=8192, help='Vocabulary size')
    parser.add_argument('--max_input_len', type=int, default=512, help='Max input length')
    parser.add_argument('--max_num_hazard_codes', type=int, default=30, help='Max number of hazard codes per example')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')

    args = parser.parse_args()
    
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=args.output_dir)
    
    logging.info(f"Loading GHS map from {args.ghs_path}")
    with open(args.ghs_path, 'r') as f:
        ghs_map = json.load(f)

    # 1. Load Data
    logging.info(f"Loading data from {args.data_path}")
    all_examples = load_data(args.data_path, ghs_map=ghs_map)
    
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
    config = model_lib.PyTorchModelConfig(
        encoder_vocab=encoder_vocab,
        decoder_vocab=decoder_vocab,
        max_input_len=args.max_input_len,
        max_num_objs=1,
        architecture_kwargs={
            'd_model': args.d_model, 
            'num_encoder_layers': args.num_encoder_layers, 
            'num_decoder_layers': args.num_decoder_layers,
        }
    )
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
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
    logging.info("Starting training...")
    steps_per_epoch = len(train_examples) // args.batch_size
    train_dl = trainer.train_dl
    
    global_step = 0

    logging.info("Running initial evaluation on validation set...")
    _, _, f1 = evaluate_model(model, val_examples, batch_size=args.batch_size, writer=writer, step=global_step)
    best_f1 = f1

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

        _, _, f1 = evaluate_model(model, val_examples, batch_size=args.batch_size, writer=writer, step=global_step)
            
        # Save Checkpoint
        if f1 > best_f1:
            best_f1 = f1
            checkpoint_path = os.path.join(args.output_dir, f"best_checkpoint.pt")
            trainer.save_checkpoint(checkpoint_path)
            logging.info(f"  Saved best checkpoint to {checkpoint_path} at epoch {epoch+1}")

    # 6. Evaluation
    logging.info("Running final evaluation on validation set...")
    evaluate_model(model, val_examples, batch_size=args.batch_size, writer=writer, step=global_step)
    
    writer.close()
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
