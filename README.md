# ADL HW3
2024 NTU ADL HW3
QLoRa finetuning on `zake7749/gemma-2-2b-it-chinese-kyara-dpo`

## Training
```
bash train.sh
```
* All the output checkpoints will be located at `--output_dir`

## Evaluation And Plot
```
bash eval_all.sh
```
* This command will evaluate all the checkpoints in `--peft_path` folder and plot the learning curve.
* Ths learning curve image would be located at  `--peft_path` and named `ppl_curve.png`.

## Inference
```
bash run.sh /path/to/base_model /path/to/lora_model /path/to/testing_data /path/to/output_file
```
* for example, if your checkpoint is at `output/checkpoint-1000`, you can run : 
```
bash run.sh zake7749/gemma-2-2b-it-chinese-kyara-dpo output/checkpoint-1000 data/public_test.json output/pred.json
```