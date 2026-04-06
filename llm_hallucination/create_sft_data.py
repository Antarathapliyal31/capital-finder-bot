import numpy
import math
import torch

from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
import json

def prompt_tokenize(prompt):
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    return prompt_tokens

def check_answer(model_ans,correct_ans):
    model_ans=model_ans.lower().strip()
    model_ans=model_ans.split("\n")[0].strip()
    for i in correct_ans:
        if model_ans.split(".")[0].strip() in i or i in model_ans:
            return 1
    return 0


device=torch.device("cpu")
dtype=torch.float32
model_name="microsoft/phi-2"

tokenizer=AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token

model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=dtype)

dataset=load_dataset("trivia_qa", "unfiltered", split="train")
sft_questions=dataset.select(range(2000))
sft_dataset=[]

for i,j in enumerate(sft_questions):
    question=j["question"]
    prompt=f"Question: {question},\n Answer:"
    text_input=prompt_tokenize(prompt)
    ground_truth_aliases=j["answer"]["aliases"]
    with torch.no_grad():
        output=model.generate(
            **text_input,output_scores=False,max_new_tokens=20,return_dict_in_generate=True,do_sample=False)

    generated_ids=output.sequences[0,text_input.input_ids.shape[1]:]
    generated_text=tokenizer.decode(generated_ids,skip_special_tokens=True)

    clean_ans=generated_text.split("\n")[0].strip()
    correct=check_answer(generated_text, ground_truth_aliases)
    if correct==1:
        confidence=0.8
    else:
        confidence=0.2

    sft_example={
        "prompt":prompt,
        "target":f"{clean_ans} [Conf:{confidence:.2f}]"
    }
    sft_dataset.append(sft_example)

    if i%100==0:
        print(f"Processed {i}/2000 | Answer: {clean_ans[:40]} | "
              f"Correct: {correct} | Target conf: {confidence}")

with open("sft_dataset.json","w") as f:
    json.dump(sft_dataset,f,indent=2)

print(f"\nDone, Created {len(sft_dataset)} SFT examples")
print(f"Correct answers: {sum(1 for x in sft_dataset if '0.80' in x["target"])}")
print(f"Wrong answers: {sum(1 for x in sft_dataset if '0.20' in x["target"])}")



