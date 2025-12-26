import json
import logging
from ordered_set import OrderedSet
import pathlib
from tqdm import tqdm
from typing import List

import numpy as np
import torch

from regress_lm import core
from regress_lm.pytorch import data_utils
from regress_lm.pytorch import model as model_lib


def preprocess_ghs_example(item, ghs_map, keys_map, add_dosage: bool = False, add_auxiliary_features: bool = False):
    """
    Custom preprocessing for PubChem/GHS task.
    x: ALL features EXCLUDING 'GHS Codes' + 'Dosage'.
    y: 'Hazards' list from 'GHS Codes', joined as string.
    """
    # 1. Extract Hazards
    ghs_codes = item.get('GHS Codes', {})
    hazards = ghs_codes.get('Hazards', [])
    
    # 2. Construct X
    # Reorder keys: SMILES, Dosage, then others
    x_obj = {}
    
    if 'SMILES' in item:
        x_obj['SMILES'] = item['SMILES']
    else:
        raise ValueError("Missing 'SMILES' key in input data")

    if add_dosage:
        # Dosage is list of Categories for each Hazard code.
        dosages = []
        for h_code in hazards:
            info = ghs_map.get(h_code)
            if info:
                dosages.append(info.get('category', 'Unknown'))
            else:
                dosages.append('Unknown')
        x_obj['Dosage'] = dosages
    
    # Add remaining keys
    if add_auxiliary_features:
        for k, v in item.items():
            if k in keys_map['non_hazard_keys'] and k != 'SMILES':
                x_obj[k] = v
    
    x_str = json.dumps(x_obj)
    
    # 3. Construct Y
    # Return list of hazard codes
    # If using HazardCodeTokenizer, y should be List[str]
    if len(hazards) == 0:
        hazards = ['NULL']
    hazards = sorted(hazards)
    hazards.append('STOP')
    y_str = [' '.join(hazards)]
    
    return x_str, y_str


def load_data(path: str, ghs_map: dict[str, str], keys_map: dict[str, str], add_dosage: bool = False, add_auxiliary_features: bool = False) -> List[core.Example]:
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
                    x_val, y_val = preprocess_ghs_example(item, ghs_map, keys_map, add_dosage, add_auxiliary_features)
                    examples.append(core.Example(x=x_val, y=y_val)) 
                except Exception as e:
                    logging.error(f"Failed to process item: {item}")
                    logging.error(f"Error: {e}")
                    pass
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .json")
        
    logging.info(f"Loaded {len(examples)} examples from {path}")
    return examples


def best_of_n_vote(preds: List[str]) -> OrderedSet[str]:
    pred = " ".join(preds)
    pred = [c for c in pred.split() if c != 'STOP']
    unique_codes, counts = np.unique(pred, return_counts=True)
    sorted_codes = [c for c, _ in sorted(zip(unique_codes, counts), key=lambda x: x[1], reverse=True)]
    if sorted_codes[0] == 'NULL':
        sorted_codes = ['NULL']
    else:
        if 'NULL' in sorted_codes:
            sorted_codes.remove('NULL')
        num_pred_codes = min(len(pred)//num_samples, len(sorted_codes))
        sorted_codes = sorted_codes[:num_pred_codes]
    return OrderedSet(sorted_codes)


def unwrap_output_objs(pred: str) -> List[str]:
    return OrderedSet([c for c in pred.split() if c != 'STOP'])
    

def compute_instance_metrics(args: tuple[core.Example, OrderedSet[str]]):
    ex, pred = args
    pred_set = pred
    gold_set = OrderedSet(ex.y[0].split()[:-1])

    tp = len(gold_set.intersection(pred_set))
    prec = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    rec = tp / len(gold_set) if len(gold_set) > 0 else 0.0
    return prec, rec


def evaluate_model(model: model_lib.PyTorchModel, examples: List[core.Example], batch_size=16, num_samples=1, temperature=0.0, writer=None, step: int = 0):
    model.eval()
    logging.info(f"Evaluating at step {step}...")
    
    total_precision = 0.0
    total_recall = 0.0
    count = 0
    
    ds = data_utils.ExampleDataset(examples)
    dl = torch.utils.data.DataLoader(
        ds, 
        batch_size=batch_size, 
        collate_fn=model.converter.convert_inputs,
        shuffle=False
    )

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dl)):
            # batch is dict. decode returns (ids, output_objs)
            # output_objs: (B, num_samples, max_num_objs)
            _, output_objs = model.decode(batch, num_samples=num_samples, temperature=temperature)
            
            start_idx = i * batch_size
            current_batch_size = output_objs.shape[0]
            batch_examples = examples[start_idx : start_idx + current_batch_size]

            output_objs = output_objs.squeeze(-1) # (B, num_samples)
            if num_samples > 1:
                output_objs = list(map(best_of_n_vote, output_objs)) # (B)
            else:
                output_objs = list(map(unwrap_output_objs, output_objs.squeeze(-1)))
            batch_results = list(map(compute_instance_metrics, zip(batch_examples, output_objs)))
            
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

