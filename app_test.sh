#!/bin/bash
python3 test_rlm.py \
    --checkpoint_path debug_fix_scheduler_test/best_checkpoint.pt \
    --data_path pubchem.compound.dedup.json \
    --ghs_path ghs_hazard_statements.json \
    --keys_path keys_classification.json \
    --add_dosage \
    --vocab_path debug_fix_scheduler_test/sentencepiece.model \
    --max_input_len 128 \
    --encoder_type vanilla \
    --d_model 512 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --batch_size 256 \
    --num_samples 1 \
    --temperature 0.0 \
    --gpu
