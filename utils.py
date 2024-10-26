from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是一位語言轉換助理，能夠根據用戶的輸入，自動判斷句子是白話文還是文言文、並且進行互譯。無論用戶提供的是白話文還是文言文，你都要進行正確的轉換並給出簡潔的翻譯。\nUSER：{instruction}\nASSISTANT："
    
    # Few-Shot 1 examples
#     return f"""你是一位語言轉換助理，能夠根據用戶的輸入，自動判斷句子是白話文還是文言文、並且進行互譯。無論用戶提供的是白話文還是文言文，你都要進行正確的轉換並給出簡潔的翻譯。
# USER：沒過十天，鮑泉果然被拘捕。\n幫我把這句話翻譯成文言文
# ASSISTANT：後未旬，果見囚執。

# USER：{instruction}\nASSISTANT："""

    # Few-Shot 2 examples
#     return f"""你是一位語言轉換助理，能夠根據用戶的輸入，自動判斷句子是白話文還是文言文、並且進行互譯。無論用戶提供的是白話文還是文言文，你都要進行正確的轉換並給出簡潔的翻譯。
# USER：沒過十天，鮑泉果然被拘捕。\n幫我把這句話翻譯成文言文
# ASSISTANT：後未旬，果見囚執。

# USER：辛未，命吳堅為左丞相兼樞密使，常楙參知政事。\n把這句話翻譯成現代文。
# ASSISTANT：初五，命令吳堅為左承相兼樞密使，常增為參知政事。

# USER：{instruction}\nASSISTANT："""

    # Few-Shot 4 examples
#     return f"""你是一位語言轉換助理，能夠根據用戶的輸入，自動判斷句子是白話文還是文言文、並且進行互譯。無論用戶提供的是白話文還是文言文，你都要進行正確的轉換並給出簡潔的翻譯。
# USER：沒過十天，鮑泉果然被拘捕。\n幫我把這句話翻譯成文言文
# ASSISTANT：後未旬，果見囚執。

# USER：辛未，命吳堅為左丞相兼樞密使，常楙參知政事。\n把這句話翻譯成現代文。
# ASSISTANT：初五，命令吳堅為左承相兼樞密使，常增為參知政事。

# USER：文言文翻譯：\n明日，趙用賢疏入。
# ASSISTANT：第二天，趙用賢的疏奏上。

# USER：翻譯成現代文：\n州民鄭五醜構逆，與叛羌傍乞鐵匆相應，令剛往鎮之。\n答案：
# ASSISTANT：渭州人鄭五醜造反，與叛逆羌傍乞鐵忽互相呼應。下令趟剛前往鎮壓。

# USER：{instruction}\nASSISTANT："""


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
