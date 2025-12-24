import argparse
import json
import logging
import random

import numpy as np
import torch

from regress_lm import vocabs
from regress_lm import tokenizers
from regress_lm.pytorch import encoders
from regress_lm.pytorch import model as model_lib

from utils import load_data, evaluate_model


def main():
    parser = argparse.ArgumentParser(description='Test RegressLM')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--ghs_path', type=str, required=True, help='Path to ghs_hazard_statements.json')
    parser.add_argument('--keys_path', type=str, required=True, help='Path to feature keys map (filter_feature.py output)')
    parser.add_argument('--add_auxiliary_features', action='store_true', help='Include auxiliary features')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to sentencepiece model')
    
    # Model Config Args (Must match training)
    parser.add_argument('--max_input_len', type=int, default=512, help='Max input length')
    parser.add_argument('--max_num_hazard_codes', type=int, default=30, help='Max number of hazard codes')
    parser.add_argument('--encoder_type', type=str, default='vanilla', choices=['vanilla', 't5gemma'], help='Encoder type')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
    
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature')
    
    args = parser.parse_args()
    
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load GHS Map
    logging.info(f"Loading GHS map from {args.ghs_path}")
    with open(args.ghs_path, 'r') as f:
        ghs_map = json.load(f)

    # Load Keys Map
    logging.info(f"Loading keys map from {args.keys_path}")
    with open(args.keys_path, 'r') as f:
        keys_map = json.load(f)
        
    # Load Data
    logging.info(f"Loading data from {args.data_path}")
    all_examples = load_data(args.data_path, ghs_map=ghs_map, keys_map=keys_map, add_auxiliary_features=args.add_auxiliary_features)
    logging.info(f"Splitting data with seed {args.seed}...")
    rng = random.Random(args.seed)
    rng.shuffle(all_examples)
    
    split_idx = int(len(all_examples) * 0.8)
    val_examples = all_examples[split_idx:]
    
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
    
    # Model Configuration
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
    
    # Initialize Model
    model = config.make_model(compile_model=False)
    model.to(device)
    
    # Load Checkpoint
    logging.info(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    
    # Run Evaluation
    evaluate_model(model, val_examples, batch_size=args.batch_size, temperature=args.temperature)

if __name__ == '__main__':
    main()
