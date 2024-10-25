import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse


def generate_outputs(model, tokenizer, data, max_length=2048):
    generated_data = []
    instructions = [get_prompt(x["instruction"]) for x in data]

    # Tokenize instructions
    tokenized_instructions = tokenizer(instructions, return_tensors="pt", padding=True)

    for i, item in enumerate(tqdm(data)):
        input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0).to(model.device)
        attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0).to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_length=max_length,
                eos_token_id=tokenizer.eos_token_id
            )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_data.append({
            "id": item["id"],
            "output": output_text
        })

    return generated_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat."
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        required=True,
        help="Path to the saved PEFT checkpoint."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to test data (.json format)."
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        required=True,
        help="Path to save output file (.json format)."
    )
    args = parser.parse_args()

    bnb_config = get_bnb_config()

    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
        revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = PeftModel.from_pretrained(model, args.peft_path)

    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    model.eval()
    generated_data = generate_outputs(model, tokenizer, data)

    with open(args.output_file_path, "w") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)

    print(f"Output saved to {args.output_file_path}")
