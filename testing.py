import time

import pandas as pd
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from torch.utils.data import Dataset

if __name__ == '__main__':
    print("Executing...")
    start_time = time.time()
    results = []

    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2", device_map="auto",
                                                 torch_dtype="auto")

    pipe = pipeline("text-generation", model=model, tokenizer=model_id, max_new_tokens=256, num_return_sequences=1,
                    return_full_text=False, device_map="auto", torch_dtype="auto")

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        {"role": "assistant",
         "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
        {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    chat_messages = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


    class MyDataset(Dataset):
        def __len__(self):
            return 600

        def __getitem__(self, i):
            return chat_messages


    dataset = MyDataset()

    for batch_size in [128]:
        print("-" * 30)
        print(f"Streaming batch_size={batch_size}")
        ind = True
        for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
            if ind:
                print('RESULT: ', out)
                ind = False