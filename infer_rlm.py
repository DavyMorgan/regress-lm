import argparse
import json
import logging

import torch

from regress_lm import vocabs
from regress_lm import tokenizers
from regress_lm.pytorch import encoders
from regress_lm.pytorch import model as model_lib
from regress_lm import core

from utils import preprocess_ghs_example, best_of_n_vote, unwrap_output_objs


def infer_example(model, example: core.Example, num_samples: int, temperature: float):
    model.eval()
    batch = model.converter.convert_inputs([example])
    
    with torch.no_grad():
        # batch is dict. decode returns (ids, output_objs)
        _, output_objs = model.decode(batch, num_samples=num_samples, temperature=temperature)
        # output_objs: (B, num_samples, max_num_objs)
        output_objs = output_objs[0, :, 0]
        if num_samples > 1:
            output_objs = best_of_n_vote(output_objs)
        else:
            output_objs = unwrap_output_objs(output_objs[0])
        return list(output_objs)


def main():
    parser = argparse.ArgumentParser(description='Inference RegressLM')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='pubchem.compound.dedup.json', help='Path to data')
    parser.add_argument('--ghs_path', type=str, default='ghs_hazard_statements.json', help='Path to ghs_hazard_statements.json')
    parser.add_argument('--keys_path', type=str, default='keys_classification.json', help='Path to keys_classification.json')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to sentencepiece model')
    parser.add_argument('--input_json', type=str, help='JSON string or path to json file representing the molecule')
    parser.add_argument('--smiles', type=str, help='SMILES string (if input_json not provided)')
    parser.add_argument('--add_dosage', action='store_true', help='Add dosage to input')
    parser.add_argument('--add_auxiliary_features', action='store_true', help='Include auxiliary features')

    # Model Config Args
    parser.add_argument('--max_input_len', type=int, default=512, help='Max input length')
    parser.add_argument('--max_num_hazard_codes', type=int, default=30, help='Max number of hazard codes')
    parser.add_argument('--encoder_type', type=str, default='vanilla', choices=['vanilla', 't5gemma'], help='Encoder type')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
    
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_samples', type=int, default=5, help='Number of generated responses per example')
    parser.add_argument('--temperature', type=float, default=0.1)

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load Data
    logging.info(f"Loading data from {args.data_path}")
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    logging.info(f"Loading GHS map from {args.ghs_path}")
    with open(args.ghs_path, 'r') as f:
        ghs_map = json.load(f)
    logging.info(f"Loading keys map from {args.keys_path}")
    with open(args.keys_path, 'r') as f:
        keys_map = json.load(f)

    # Process Input
    item = {}
    if args.input_json:
        if args.input_json.endswith('.json'):
            with open(args.input_json, 'r') as f:
                item = json.load(f)
        else:
            item = json.loads(args.input_json)
        assert 'SMILES' in item
    elif args.smiles:
        item = {'SMILES': args.smiles}
    else:
        raise ValueError("Must provide --input_json or --smiles")

    smiles = item['SMILES']
    data_in_db = next((d for d in data if d['SMILES'] == smiles), None)
    if data_in_db is not None:
        item = data_in_db
    else:
        logging.warning(f"SMILES {smiles} not found in data. Using input as is.")

    logging.info("Preprocessing input...")
    # process_ghs_example returns x_str, y_str. We ignore y_str.
    x_str, y_str = preprocess_ghs_example(item, ghs_map, keys_map, add_dosage=args.add_dosage, add_auxiliary_features=args.add_auxiliary_features)
    logging.info(f"Input string: {x_str}")
    logging.info(f"Label string: {y_str}")

    # Load Vocabs
    logging.info(f"Loading encoder vocabulary from {args.vocab_path}")
    encoder_vocab = vocabs.SentencePieceVocab(args.vocab_path)
    
    logging.info("Using HazardCodeTokenizer for Decoder")
    decoder_vocab = vocabs.DecoderVocab(
        tokenizers.HazardCodeTokenizer(
            all_hazard_codes=list(ghs_map.keys())+['NULL', 'STOP'],
            max_num_hazard_codes=args.max_num_hazard_codes,
        )
    )

    # Model Config
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

    # Load Model
    logging.info("Initializing model...")
    model = config.make_model(compile_model=False)
    model.to(device)
    
    model_path = args.checkpoint_path
    logging.info(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])

    # Inference
    logging.info("Running inference...")
    example = core.Example(x=x_str, y="") # Y is dummy
    output = infer_example(model, example, num_samples=args.num_samples, temperature=args.temperature)
    
    logging.info(f"Predicted Hazards: {output}")
    logging.info(f"Label: {list(unwrap_output_objs(y_str[0]))}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
