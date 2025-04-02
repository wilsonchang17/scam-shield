import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 請根據需求修改模型與 adapter 的路徑
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_MODEL_PATH = "models/checkpoint-llama8b"

def load_model():
    # 載入 tokenizer 與模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",         # 自動選擇設備
        torch_dtype=torch.float16  # 使用 FP16 加速
    )
    # 載入 adapter，整合到模型上
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL_PATH)
    return tokenizer, model

def chat(tokenizer, model):
    print("開始與 Llama 對話 (輸入 exit 或 quit 結束對話)")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # 建立一個簡單的對話 prompt
        prompt = f"User: {user_input}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 使用模型生成回答，可依需求調整參數
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,    # 使用取樣生成回答
            top_p=0.95,
            temperature=0.8
        )
        output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        # 嘗試擷取 Assistant 回答部分
        response = output_text.split("Assistant:")[-1].strip()
        print(f"Assistant: {response}\n")

if __name__ == "__main__":
    tokenizer, model = load_model()
    chat(tokenizer, model)