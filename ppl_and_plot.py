import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse
import os
import re
import matplotlib.pyplot as plt

def perplexity(
    model, tokenizer, data, max_length=2048,
):
    data_size = len(data)
    instructions = [get_prompt(x["instruction"]) for x in data]
    outputs = [x["output"] for x in data]

    # Tokenize data
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
    tokenized_outputs = tokenizer(outputs, add_special_tokens=False)
    output_masks = []

    # Format data
    for i in range(data_size):
        instruction_input_ids = [tokenizer.bos_token_id] + \
            tokenized_instructions["input_ids"][i]
        output_input_ids = tokenized_outputs["input_ids"][i] + \
            [tokenizer.eos_token_id]
        tokenized_instructions["input_ids"][i] = instruction_input_ids + \
            output_input_ids
        tokenized_instructions["attention_mask"][i] = [
            1] * len(tokenized_instructions["input_ids"][i])
        output_mask = [0] * len(instruction_input_ids) + \
            [1] * len(output_input_ids)

        tokenized_instructions["input_ids"][i] = torch.tensor(
            tokenized_instructions["input_ids"][i][:max_length])
        tokenized_instructions["attention_mask"][i] = torch.tensor(
            tokenized_instructions["attention_mask"][i][:max_length])
        output_mask = torch.tensor(output_mask[:max_length])
        output_masks.append(output_mask)

    # Calculate ppl
    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in tqdm(range(data_size)):
        input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0)
        attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0)
        output_mask = output_masks[i].unsqueeze(0)
        label = input_ids

        with torch.no_grad():
            out_logits = model(input_ids, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_label = label[..., 1:].contiguous()
        shift_output_mask = output_mask[..., 1:].contiguous()
        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2),
             shift_label) * shift_output_mask).sum(1)
            / shift_output_mask.sum(1)
        )
        ppls += perplexity_batch.tolist()
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

def plot_ppl(ppl_list, output_path):
    # Extract steps and mean PPL values from ppl_list
    steps = [item['step'] for item in ppl_list]
    mean_ppl = [item['mean_ppl'] for item in ppl_list]

    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(steps, mean_ppl, marker='o', color='b', linestyle='-', linewidth=2, markersize=5)
    
    # Adding labels and title
    plt.xlabel('Step')
    plt.ylabel('PPL Score')
    plt.title('Learning Curve (PPL over Steps)')
    plt.grid(True)

    plt.savefig(output_path + "/ppl_curve.png")
    plt.close()

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
        help="Path to the folder containing multiple PEFT checkpoints."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    args = parser.parse_args()

    # Load model
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

    # Load test data
    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    # List all checkpoints in peft_path
    checkpoint_folders = [f for f in os.listdir(args.peft_path) if re.match(r'checkpoint-\d+', f)]
    checkpoint_folders.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # Sort by step number

    ppl_list = []

    # Iterate through each checkpoint folder
    for checkpoint_folder in checkpoint_folders:
        step = int(re.findall(r'\d+', checkpoint_folder)[0])
        checkpoint_path = os.path.join(args.peft_path, checkpoint_folder)

        # Load LoRA for each checkpoint
        model_peft = PeftModel.from_pretrained(model, checkpoint_path)
        model_peft.eval()

        # Calculate perplexity for the current checkpoint
        ppl = perplexity(model_peft, tokenizer, data)
        print(f"Checkpoint {step}: Mean perplexity = {ppl['mean_perplexity']}")

        # Append result to list
        ppl_list.append({"step": step, "mean_ppl": ppl["mean_perplexity"]})

    # Plot the results
    plot_ppl(ppl_list, args.peft_path)
