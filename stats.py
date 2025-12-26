import argparse
from collections import Counter
import decimal
import ijson
import json
import numpy as np
import sys
from tqdm import tqdm


def default_encoder(o):
    if isinstance(o, decimal.Decimal):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(description="Extract unique GHS Hazard codes from PubChem JSON")
    parser.add_argument('--path', required=True, help="Path to pubchem json file")
    args = parser.parse_args()
    
    # Pass 1: Count occurrences
    print(f"Scanning {args.path} to count SMILES occurrences...")
    smiles_counts = Counter()
    
    try:
        with open(args.path, 'rb') as f:
            # Assumes the file is a list of objects. 'item' parses each object in the array.
            items = ijson.items(f, 'item')
            for item in tqdm(items, desc="Counting"):
                smiles = item.get('SMILES')
                if smiles:
                    smiles_counts[smiles] += 1
    except FileNotFoundError:
        print(f"Error: File {args.path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file during count pass: {e}")
        sys.exit(1)
        
    print(f"Index built. Found {len(smiles_counts)} unique SMILES among total items.")

    smiles_length = [len(smiles) for smiles in smiles_counts.keys()]
    print(f"Min SMILES length: {min(smiles_length)}")
    print(f"Max SMILES length: {max(smiles_length)}")
    print(f"Avg SMILES length: {sum(smiles_length) / len(smiles_length)}")
    print(f"Percentiles of SMILES length at 10, 25, 50, 75, 90: {np.percentile(smiles_length, [10, 25, 50, 75, 90])}")

    # Pass 2: Filter and Write
    output_file = "pubchem.compound.dedup.json"
    print(f"Filtering unique items to {output_file}...")
    
    unique_hazards = set()
    num_items_kept = 0
    num_hazards = 0
    max_hazards = 0
    
    try:
        with open(args.path, 'rb') as f_in, open(output_file, 'w') as f_out:
            f_out.write('[\n')
            
            items = ijson.items(f_in, 'item')
            first_item = True
            
            for item in tqdm(items, desc="Filtering"):
                smiles = item.get('SMILES')
                if smiles and smiles_counts[smiles] == 1:
                    # Write item
                    if not first_item:
                        f_out.write(',\n')
                    json.dump(item, f_out, default=default_encoder)
                    first_item = False
                    
                    # Update stats
                    num_items_kept += 1
                    ghs = item.get('GHS Codes', {})
                    hazards = ghs.get('Hazards', [])
                    
                    if len(hazards) > 0:
                        num_hazards += 1
                        for h in hazards:
                            if isinstance(h, str):
                                unique_hazards.add(h)
                    
                    max_hazards = max(max_hazards, len(hazards))

            f_out.write('\n]\n')
            
    except Exception as e:
        print(f"Error processing file during filter pass: {e}")
        sys.exit(1)

    print(f"\nProcessing complete.")
    print(f"Kept {num_items_kept} items (discarded duplicates).")
    print(f"Found {len(unique_hazards)} unique hazard codes in filtered dataset.")
    print(f"Found {num_hazards} total hazards in filtered dataset.")
    sorted_hazards = sorted(list(unique_hazards))
    print(f"Hazards: {sorted_hazards}")
    print(f"Max hazards per item: {max_hazards}")
    
    
if __name__ == '__main__':
    main()
