# llm_classifier.py
import os
import re
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# 避免 tokenizers 並行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 設定模型參數（請依照實際環境修改路徑）
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_MODEL_PATH = "models/checkpoint-llama8b"   # 請修改成你的 adapter checkpoint 路徑
#MODEL_PATH = "/home/wilsonchang/model"        # 請修改成你的模型檔案路徑
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# 載入 tokenizer 與模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    #MODEL_PATH,
    model_id,
    #torch_dtype=torch.float16,
    #device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER_MODEL_PATH)

def classify_message_llm(message):
    """
    使用 LLaMA 8B 模型判斷訊息是否為 scam
    回傳 1 表示 scam，0 表示 not scam
    "Do not provide any additional explanation, and do not assume anything beforehand.\n"
    """
    print("message:", message)
    prompt = (
        "Is the following message scam? Please answer only 'Yes' or 'No'.\n"
        f"Message: {message}\n"
        "Answer:"
    )
    try:
        # Tokenize 輸入 prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
                top_k=1
            )
        # 取得生成結果中超過輸入部分的 token
        generated_tokens = outputs[0][inputs['input_ids'].size(1):]
        full_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()
        # 利用正則表達式找出 "yes" 或 "no"
        matches = re.findall(r'\b(yes|no)\b', full_answer)
        if matches:
            final_decision = matches[-1]
            if final_decision == 'yes':
                return 1  # scam
            elif final_decision == 'no':
                return 0  # not scam
        else:
            # 若無有效答案，預設回傳 scam (1)
            return -1
    except Exception as e:
        print(f"LLM 判斷錯誤: {e}")
        return -1  # 若發生錯誤則回傳 -1

if __name__ == '__main__':
    # 測試 LLM 判斷函式
    test_message = "Hello, how are you."
    result = classify_message_llm(test_message)
    print(f"LLM 判斷結果: {result}  (1: scam, 0: not scam)")