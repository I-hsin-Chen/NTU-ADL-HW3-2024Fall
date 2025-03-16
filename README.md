# 2024 NTU ADL HW3
This repository contains coursework from National Taiwan University CSIE5431: Applied Deep Learning.

* Task: QLoRa fine-tuning on `zake7749/gemma-2-2b-it-chinese-kyara-dpo`
* Student: I-Hsin Chen
* Affiliation: Graduate Institute of Networking and Multimedia, National Taiwan University

## Environment Setup
* Install torch depending on your CUDA version: `pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118`
* Install packages with allowed versions according to spec: `pip install transformers==4.45.1 bitsandbytes==0.44.1 peft==0.13.0`
* Install some other required packages such as `sentencepiece` and `matplotlib`.

## Training
```
bash train.sh
```
* All the output checkpoints will be located at `--output_dir`

## Evaluation And Plot
```
bash eval_all.sh
```
* This command will evaluate all checkpoints in the folder specified by --peft_path and plot the learning curve.
* The generated image (ppl_curve.png) will be saved in the same folder.

## Inference
```
bash run.sh /path/to/base_model /path/to/lora_model /path/to/testing_data /path/to/output_file
```
* for example, if your checkpoint is at `output/checkpoint-1000`, you can run : 
```
bash run.sh zake7749/gemma-2-2b-it-chinese-kyara-dpo output/checkpoint-1000 data/public_test.json output/pred.json
```
