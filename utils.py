from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是一位語言轉換助理，能夠根據用戶的輸入，自動判斷句子是白話文還是文言文、並且進行互譯。無論用戶提供的是白話文還是文言文，你都要進行正確的轉換並給出簡潔的翻譯。\nUSER：{instruction}\nASSISTANT："


def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    return quantization_config
