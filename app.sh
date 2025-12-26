#!/bin/bash
python3 train_rlm.py \
    --data_path pubchem.compound.dedup.json \
    --ghs_path ghs_hazard_statements.json \
    --keys_path keys_classification.json \
    --output_dir debug_fix_scheduler_slim_test \
    --vocab_size 2048 \
    --max_input_len 256 \
    --train_batch_size 128 \
    --eval_batch_size 512 \
    --epochs 100 \
    --lr 0.0001 \
    --encoder_type vanilla \
    --d_model 512 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --num_samples 5 \
    --temperature 0.1 \
    --gpu
