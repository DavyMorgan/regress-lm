#!/bin/bash
. /venv/main/bin/activate
python3 infer_rlm.py \
    --checkpoint_path debug_test/best_checkpoint.pt \
    --vocab_path debug_test/sentencepiece.model \
    --ghs_path ghs_hazard_statements.json \
    --keys_path keys_classification.json \
    --smiles "CCO" \
    --encoder_type vanilla \
    --d_model 64 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --gpu
