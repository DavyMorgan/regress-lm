import ijson
import argparse
import sys
import json
from tqdm import tqdm

def contains_hazard(value):
    """
    Recursively check if 'hazard' appears in the value (string, list, or dict).
    """
    if isinstance(value, str):
        return "hazard" in value.lower()
    elif isinstance(value, list):
        return any(contains_hazard(x) for x in value)
    elif isinstance(value, dict):
        # Check both keys and values
        return any(contains_hazard(k) or contains_hazard(v) for k, v in value.items())
    return False

def main():
    parser = argparse.ArgumentParser(description="Find keys unrelated to hazard info")
    parser.add_argument('--path', required=True, help="Path to pubchem json file")
    parser.add_argument('--output', help="Path to output json file to save keys")
    args = parser.parse_args()

    all_keys = set()
    hazard_keys = set()
    
    print(f"Scanning {args.path}...")
    
    try:
        with open(args.path, 'rb') as f:
            items = ijson.items(f, 'item')
            for item in tqdm(items):
                current_keys = item.keys()
                all_keys.update(current_keys)
                
                for key in current_keys:
                    # Optimization: if we already know it's a hazard key, skip checking
                    if key in hazard_keys:
                        continue
                        
                    # Also check if the KEY itself contains "hazard"
                    if "hazard" in key.lower() or contains_hazard(item[key]):
                        hazard_keys.add(key)
                        
    except FileNotFoundError:
        print(f"Error: File {args.path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

    non_hazard_keys = sorted(list(all_keys - hazard_keys))
    
    print(f"Found {len(all_keys)} total keys.")
    print(f"Found {len(hazard_keys)} hazard keys.")
    print(f"Found {len(non_hazard_keys)} keys unrelated to hazard information.")
    
    print("\nKeys unrelated to hazard information:")
    for key in non_hazard_keys:
        print(key)

    if args.output:
        output_data = {
            "all_keys": sorted(list(all_keys)),
            "hazard_keys": sorted(list(hazard_keys)),
            "non_hazard_keys": non_hazard_keys
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"\nSaved keys classification to {args.output}")

if __name__ == '__main__':
    main()
