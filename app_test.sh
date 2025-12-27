#!/bin/bash
python3 test_rlm.py \
    --checkpoint_path debug_fix_scheduler_slim_test/best_checkpoint.pt \
    --data_path pubchem.compound.dedup.json \
    --ghs_path ghs_hazard_statements.json \
    --keys_path keys_classification.json \
    --vocab_path debug_fix_scheduler_slim_test/sentencepiece.model \
    --max_input_len 256 \
    --encoder_type vanilla \
    --d_model 512 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --eval_batch_size 2048 \
    --num_samples 5 \
    --temperature 0.1 \
    --gpu
