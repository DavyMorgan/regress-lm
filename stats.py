import ijson
import argparse
import sys
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Extract unique GHS Hazard codes from PubChem JSON")
    parser.add_argument('--path', required=True, help="Path to pubchem json file")
    args = parser.parse_args()
    
    unique_hazards = set()
    
    print(f"Scanning {args.path} for hazards...")
    
    num_items = 0
    num_hazards = 0
    try:
        with open(args.path, 'rb') as f:
            # Assumes the file is a list of objects. 'item' parses each object in the array.
            items = ijson.items(f, 'item')
            
            for item in tqdm(items):
                ghs = item.get('GHS Codes', {})
                hazards = ghs.get('Hazards', [])

                if len(hazards) > 0:
                    num_hazards += 1
                    for h in hazards:
                        if isinstance(h, str):
                            unique_hazards.add(h)
                num_items += 1
    except FileNotFoundError:
        print(f"Error: File {args.path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)
    
    print(f"\nFound {len(unique_hazards)} unique hazard codes.")
    print(f"Found {num_hazards} hazards in total.")
    print(f"Found {num_items} items in total.")
    sorted_hazards = sorted(list(unique_hazards))
    print(sorted_hazards)
    
if __name__ == '__main__':
    main()
