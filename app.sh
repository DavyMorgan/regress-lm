#!/bin/bash
python3 train_rlm.py \
    --data_path pubchem.compound.dedup.json \
    --ghs_path ghs_hazard_statements.json \
    --keys_path keys_classification.json \
    --add_auxiliary_features \
    --output_dir debug_large_aux_t5gemma_test \
    --vocab_size 8192 \
    --max_input_len 512 \
    --batch_size 128 \
    --epochs 100 \
    --encoder_type t5gemma \
    --d_model 512 \
    --num_encoder_layers 12 \
    --num_decoder_layers 12 \
    --num_samples 5 \
    --temperature 0.1 \
    --gpu
