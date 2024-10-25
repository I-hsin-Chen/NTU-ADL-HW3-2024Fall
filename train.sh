#!bin/bash
python train.py \
    --model_name_or_path zake7749/gemma-2-2b-it-chinese-kyara-dpo \
    --dataset_format custom \
    --output_dir ./output/3e-5 \
    --dataset data/train.json \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --bf16 \
    --do_train \
    --max_steps 3500 \
    --save_steps 100 \
    --learning_rate 1e-4 \
    # --lora_r 16 \
