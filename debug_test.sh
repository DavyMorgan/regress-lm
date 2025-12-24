#!/bin/bash
python3 test_rlm.py \
    --checkpoint_path debug_test/best_checkpoint.pt \
    --data_path pubchem.compound.100.json \
    --ghs_path ghs_hazard_statements.json \
    --keys_path keys_classification.json \
    --vocab_path debug_test/sentencepiece.model \
    --encoder_type vanilla \
    --d_model 64 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --batch_size 4
