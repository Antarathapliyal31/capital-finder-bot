import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# ---- Config ----
device = torch.device("cpu")
dtype = torch.float32
model_name = "microsoft/phi-2"
num_samples = 8  # sampling runs per question
temperature = 0.7
num_questions = 2000
question_offset = 2000  # use range(2000, 4000), different from SFT

# ---- Load tokenizer and SFT model ----
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
print("Loading SFT LoRA adapter...")
model = PeftModel.from_pretrained(base_model, "./sft_lora_adapter")
model.eval()
print("Model loaded.")


def check_answer(model_ans, correct_ans):
    """Check if model's answer matches any ground truth alias."""
    model_ans = model_ans.lower().strip()
    model_ans = model_ans.split("\n")[0].strip()
    for alias in correct_ans:
        alias_clean = alias.lower().strip()
        if alias_clean in model_ans or model_ans.split(".")[0].strip() in alias_clean:
            return 1
    return 0


def extract_answer_text(generated_text):
    """Extract answer text, stripping any [Conf:...] suffix."""
    text = generated_text.split("\n")[0].strip()
    # Remove [Conf:X.XX] if present
    import re
    text = re.sub(r'\s*\[Conf:\d+\.\d+\]', '', text).strip()
    return text


# ---- Load TriviaQA ----
dataset = load_dataset("trivia_qa", "unfiltered", split="train")
dpo_questions = dataset.select(range(question_offset, question_offset + num_questions))
print(f"Loaded {len(dpo_questions)} questions (indices {question_offset}-{question_offset + num_questions})")

# ---- Generate DPO pairs ----
dpo_pairs = []

for i, example in enumerate(dpo_questions):
    question = example["question"]
    aliases = example["answer"]["aliases"]
    prompt = f"Question: {question},\n Answer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Multiple sampling runs to estimate consistency
    answers = []
    correct_counts = 0

    with torch.no_grad():
        for s in range(num_samples):
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
            )
            gen_ids = output[0, inputs.input_ids.shape[1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            ans_text = extract_answer_text(gen_text)
            answers.append(ans_text)
            correct_counts += check_answer(gen_text, aliases)

    # Empirical accuracy from sampling = Brier-optimal confidence
    empirical_accuracy = correct_counts / num_samples

    # Clamp to [0.05, 0.95] to avoid extreme values
    optimal_conf = max(0.05, min(0.95, empirical_accuracy))

    # Use the most common answer (greedy-like) as the answer text
    # Do a greedy decode for the canonical answer
    with torch.no_grad():
        greedy_output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
        )
    greedy_ids = greedy_output[0, inputs.input_ids.shape[1]:]
    greedy_text = tokenizer.decode(greedy_ids, skip_special_tokens=True)
    answer_text = extract_answer_text(greedy_text)

    # Check if greedy answer is correct
    greedy_correct = check_answer(greedy_text, aliases)

    # Construct winner/loser pair
    # Winner: Brier-optimal confidence
    # Loser: mirrored confidence (far from optimal)
    loser_conf = max(0.05, min(0.95, 1.0 - optimal_conf))

    winner = f"{answer_text} [Conf:{optimal_conf:.2f}]"
    loser = f"{answer_text} [Conf:{loser_conf:.2f}]"

    dpo_pairs.append({
        "prompt": prompt,
        "winner": winner,
        "loser": loser,
    })

    if i % 50 == 0:
        print(
            f"[{i}/{num_questions}] Q: {question[:50]}... | "
            f"Ans: {answer_text[:30]} | Correct: {greedy_correct} | "
            f"Empirical acc: {empirical_accuracy:.2f} | "
            f"Winner conf: {optimal_conf:.2f} | Loser conf: {loser_conf:.2f}"
        )

# ---- Save ----
with open("dpo_pairs.json", "w") as f:
    json.dump(dpo_pairs, f, indent=2)

print(f"\nDone! Created {len(dpo_pairs)} DPO pairs.")
print(f"Saved to dpo_pairs.json")

# Stats
confs = [float(p["winner"].split("Conf:")[1].rstrip("]")) for p in dpo_pairs]
print(f"Winner confidence stats: mean={np.mean(confs):.3f}, "
      f"min={np.min(confs):.2f}, max={np.max(confs):.2f}, "
      f"std={np.std(confs):.3f}")
