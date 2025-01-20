import time

import pandas as pd
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    print("Executing...")
    start_time = time.time()
    results = []
    prompts = [
                  "What is the capital of France?",
                  "What is the capital of Germany?",
                  "What is the capital of Italy?",
                  "What is the capital of Spain?",
                  "What is the capital of Portugal?",
              ] * 200
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    weights_path = "/hf-home/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/model.safetensors.index.json"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained(model_id", pad_token_id=tokenizer.eos_token_id)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, device=0)
    # results = []
    # for prompt in tqdm(prompts):
    #     result = pipe(prompt, pad_token_id=tokenizer.eos_token_id)
    #     results.append(result)

    model = load_checkpoint_and_dispatch(
        model,
        weights_path,
        device_map="auto",
        no_split_module_classes=["LlamaDecoderLayer"], # LlamaDecoderLayer layer has a residual connection, so it should not be split
        dtype=torch.float16,
    )

    batch_size = 256
    all_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded_outputs)

    print(f"Execution time in seconds: {time.time() - start_time}")

    df = pd.DataFrame({'prompt': prompts, 'response': results})
    df.to_csv("/testing/prompt-test-results.tsv", index=False, sep="\t")