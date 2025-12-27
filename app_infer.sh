#!/bin/bash
. /venv/main/bin/activate
python3 infer_rlm.py \
    --checkpoint_path debug_fix_scheduler_smiles_dosage_test/best_checkpoint.pt \
    --vocab_path debug_fix_scheduler_smiles_dosage_test/sentencepiece.model \
    --ghs_path ghs_hazard_statements.json \
    --keys_path keys_classification.json \
    --input_json test_infer.json \
    --add_dosage \
    --encoder_type vanilla \
    --max_input_len 256 \
    --d_model 512 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --num_samples 1 \
    --temperature 0.0 \
    --gpu
