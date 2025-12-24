#!/bin/bash
. /venv/main/bin/activate

python3 test_rlm.py \
    --checkpoint_path debug_large_test/best_checkpoint.pt \
    --data_path pubchem.compound.json \
    --ghs_path ghs_hazard_statements.json \
    --keys_path keys_classification.json \
    --vocab_path debug_large_test/sentencepiece.model \
    --encoder_type vanilla \
    --d_model 512 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --batch_size 128 \
    --gpu
