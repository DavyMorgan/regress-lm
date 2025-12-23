#!/bin/bash
python3 train_rlm.py \
    --data_path pubchem.compound.100.json \
    --ghs_path ghs_hazard_statements.json \
    --output_dir debug_test \
    --vocab_size 500 \
    --epochs 2 \
    --encoder_type vanilla \
    --d_model 64 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --batch_size 4 \
    --gpu
