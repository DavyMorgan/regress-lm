#!/bin/bash
python3 train_rlm.py \
    --data_path pubchem.compound.dedup.json \
    --ghs_path ghs_hazard_statements.json \
    --keys_path keys_classification.json \
    --add_auxiliary_features \
    --output_dir debug_large_test \
    --vocab_size 2048 \
    --epochs 30 \
    --encoder_type vanilla \
    --d_model 512 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --batch_size 128 \
    --gpu
