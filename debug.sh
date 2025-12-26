#!/bin/bash
python3 train_rlm.py \
    --data_path pubchem.compound.100.json \
    --ghs_path ghs_hazard_statements.json \
    --keys_path keys_classification.json \
    --add_auxiliary_features \
    --output_dir debug_test \
    --vocab_size 500 \
    --epochs 2 \
    --encoder_type vanilla \
    --d_model 64 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --warmup_steps 10 \
    --hold_steps 10 \
    --batch_size 4 \
    --num_samples 5 \
    --temperature 0.2 \
    --gpu
