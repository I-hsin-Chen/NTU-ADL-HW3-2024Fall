#!bin/bash
python inference.py \
    --base_model_path $1 \
    --peft_path $2 \
    --test_data_path $3 \
    --output_file_path $4