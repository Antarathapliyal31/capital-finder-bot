import torch
import math
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# ---- Config ----
device = torch.device("cpu")
dtype = torch.float32
model_name = "microsoft/phi-2"
num_eval = 1000
eval_offset = 4000  # range(4000, 5000) — held-out questions
num_bins = 10

# ---- Load tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ---- Load TriviaQA ----
dataset = load_dataset("trivia_qa", "unfiltered", split="train")
eval_questions = dataset.select(range(eval_offset, eval_offset + num_eval))
print(f"Loaded {num_eval} eval questions (indices {eval_offset}-{eval_offset + num_eval})")


def check_answer(model_ans, aliases):
    model_ans = model_ans.lower().strip()
    model_ans = model_ans.split("\n")[0].strip()
    for alias in aliases:
        alias_clean = alias.lower().strip()
        if alias_clean in model_ans or model_ans.split(".")[0].strip() in alias_clean:
            return 1
    return 0


def extract_conf_from_text(text):
    """Parse [Conf:X.XX] from generated text."""
    match = re.search(r'\[Conf:(\d+\.\d+)\]', text)
    return float(match.group(1)) if match else 0.50


def compute_logprob_confidence(model, tokenizer, prompt, max_new_tokens=30):
    """Generate text and compute mean logprob confidence (for base model)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
        )
    gen_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Compute mean logprob confidence
    token_probs = []
    for step, scores in enumerate(outputs.scores):
        probs = torch.softmax(scores.float(), dim=-1)
        chosen = outputs.sequences[0, inputs.input_ids.shape[1] + step]
        token_probs.append(probs[0, chosen].item())

    valid_probs = [p for p in token_probs if p > 0]
    if valid_probs:
        log_probs = [math.log(p) for p in valid_probs]
        confidence = math.exp(sum(log_probs) / len(log_probs))
    else:
        confidence = 0.50

    return gen_text, confidence


def generate_with_conf(model, tokenizer, prompt, max_new_tokens=40):
    """Generate text and parse [Conf:X.XX] from output."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    gen_ids = output[0, inputs.input_ids.shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    confidence = extract_conf_from_text(gen_text)
    return gen_text, confidence


def compute_metrics(results):
    """Compute ECE, Brier score, overconfidence rate."""
    confidences = [r["confidence"] for r in results]
    corrects = [r["correct"] for r in results]
    n = len(results)

    # ECE
    bins = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    bin_data = []
    for b in range(num_bins):
        mask = [(c >= bins[b] and c < bins[b + 1]) for c in confidences]
        bin_corrects = [c for c, m in zip(corrects, mask) if m]
        bin_confs = [c for c, m in zip(confidences, mask) if m]
        count = len(bin_corrects)
        if count > 0:
            avg_acc = np.mean(bin_corrects)
            avg_conf = np.mean(bin_confs)
            ece += (count / n) * abs(avg_acc - avg_conf)
            bin_data.append({
                "midpoint": (bins[b] + bins[b + 1]) / 2,
                "accuracy": avg_acc,
                "avg_conf": avg_conf,
                "count": count,
            })

    # Brier score: 1 - (c - y)^2 averaged
    brier = np.mean([1 - (c - y) ** 2 for c, y in zip(confidences, corrects)])

    # Overconfidence rate: wrong answers with conf > 0.50
    wrong_high = sum(1 for c, y in zip(confidences, corrects) if y == 0 and c > 0.50)
    total_wrong = sum(1 for y in corrects if y == 0)
    overconf_rate = wrong_high / total_wrong if total_wrong > 0 else 0.0

    accuracy = np.mean(corrects)

    return {
        "ece": ece,
        "brier": brier,
        "overconfidence_rate": overconf_rate,
        "accuracy": accuracy,
        "n": n,
        "bin_data": bin_data,
    }


def run_eval(model, model_label, use_logprob=False):
    """Run evaluation on the held-out set."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_label}")
    print(f"{'='*60}")

    results = []
    for i, example in enumerate(eval_questions):
        question = example["question"]
        aliases = example["answer"]["aliases"]
        prompt = f"Question: {question},\n Answer:"

        if use_logprob:
            gen_text, confidence = compute_logprob_confidence(model, tokenizer, prompt)
        else:
            gen_text, confidence = generate_with_conf(model, tokenizer, prompt)

        correct = check_answer(gen_text, aliases)
        results.append({
            "question": question,
            "generated": gen_text.split("\n")[0].strip(),
            "confidence": confidence,
            "correct": correct,
        })

        if i % 100 == 0:
            print(f"  [{i}/{num_eval}] Ans: {gen_text[:40]}... | Conf: {confidence:.3f} | Correct: {correct}")

    metrics = compute_metrics(results)
    print(f"\n  Results for {model_label}:")
    print(f"    Accuracy:           {metrics['accuracy']:.4f}")
    print(f"    ECE:                {metrics['ece']:.4f}")
    print(f"    Brier Score:        {metrics['brier']:.4f}")
    print(f"    Overconfidence:     {metrics['overconfidence_rate']:.4f}")

    return results, metrics


def abstention_sweep(results, model_label):
    """Compute accuracy/coverage at various abstention thresholds."""
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    sweep = []
    for t in thresholds:
        answered = [r for r in results if r["confidence"] >= t]
        coverage = len(answered) / len(results)
        acc = np.mean([r["correct"] for r in answered]) if answered else 0.0
        sweep.append({"threshold": t, "coverage": coverage, "accuracy": acc})
    return sweep


# ============================================================
# Checkpoint 1: Base Phi-2 (logprob confidence)
# ============================================================
print("Loading base Phi-2...")
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
base_results, base_metrics = run_eval(base_model, "Base Phi-2", use_logprob=True)
base_sweep = abstention_sweep(base_results, "Base Phi-2")
del base_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================================
# Checkpoint 2: After SFT (verbalized confidence)
# ============================================================
print("\nLoading SFT model...")
sft_base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
sft_model = PeftModel.from_pretrained(sft_base, "./sft_lora_adapter")
sft_model.eval()
sft_results, sft_metrics = run_eval(sft_model, "SFT Model", use_logprob=False)
sft_sweep = abstention_sweep(sft_results, "SFT Model")
del sft_model, sft_base
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================================
# Checkpoint 3: After DPO (verbalized confidence)
# ============================================================
print("\nLoading DPO model...")
dpo_base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
dpo_model = PeftModel.from_pretrained(dpo_base, "./dpo_lora_adapter")
dpo_model.eval()
dpo_results, dpo_metrics = run_eval(dpo_model, "DPO Model", use_logprob=False)
dpo_sweep = abstention_sweep(dpo_results, "DPO Model")
del dpo_model, dpo_base

# ============================================================
# Save results
# ============================================================
import os
os.makedirs("results", exist_ok=True)

eval_output = {
    "base": {"metrics": {k: v for k, v in base_metrics.items() if k != "bin_data"},
             "bin_data": base_metrics["bin_data"], "sweep": base_sweep},
    "sft": {"metrics": {k: v for k, v in sft_metrics.items() if k != "bin_data"},
            "bin_data": sft_metrics["bin_data"], "sweep": sft_sweep},
    "dpo": {"metrics": {k: v for k, v in dpo_metrics.items() if k != "bin_data"},
            "bin_data": dpo_metrics["bin_data"], "sweep": dpo_sweep},
}

with open("results/evaluation_results.json", "w") as f:
    json.dump(eval_output, f, indent=2)
print("\nSaved evaluation_results.json")

# ============================================================
# Calibration Plot
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Calibration curves
ax = axes[0]
ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
for label, metrics, color in [
    ("Base Phi-2", base_metrics, "red"),
    ("SFT", sft_metrics, "blue"),
    ("DPO", dpo_metrics, "green"),
]:
    if metrics["bin_data"]:
        mids = [b["midpoint"] for b in metrics["bin_data"]]
        accs = [b["accuracy"] for b in metrics["bin_data"]]
        ax.plot(mids, accs, 'o-', color=color, label=f'{label} (ECE={metrics["ece"]:.3f})', markersize=8)
ax.set_xlabel('Confidence')
ax.set_ylabel('Accuracy')
ax.set_title('Calibration Curves')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Abstention sweep - accuracy vs coverage
ax = axes[1]
for label, sweep, color in [
    ("Base Phi-2", base_sweep, "red"),
    ("SFT", sft_sweep, "blue"),
    ("DPO", dpo_sweep, "green"),
]:
    coverages = [s["coverage"] for s in sweep]
    accuracies = [s["accuracy"] for s in sweep]
    ax.plot(coverages, accuracies, 'o-', color=color, label=label, markersize=6)
ax.set_xlabel('Coverage (fraction answered)')
ax.set_ylabel('Accuracy (on answered)')
ax.set_title('Abstention Sweep')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Summary bar chart
ax = axes[2]
labels = ['Base', 'SFT', 'DPO']
ece_vals = [base_metrics["ece"], sft_metrics["ece"], dpo_metrics["ece"]]
overconf_vals = [base_metrics["overconfidence_rate"], sft_metrics["overconfidence_rate"], dpo_metrics["overconfidence_rate"]]
x = np.arange(len(labels))
w = 0.35
ax.bar(x - w/2, ece_vals, w, label='ECE', color='steelblue')
ax.bar(x + w/2, overconf_vals, w, label='Overconfidence', color='salmon')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Rate')
ax.set_title('ECE & Overconfidence')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("results/calibration_plot.png", dpi=150, bbox_inches="tight")
print("Saved calibration_plot.png")

# ============================================================
# Final summary
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"{'Metric':<25} {'Base':>10} {'SFT':>10} {'DPO':>10}")
print("-" * 55)
print(f"{'Accuracy':<25} {base_metrics['accuracy']:>10.4f} {sft_metrics['accuracy']:>10.4f} {dpo_metrics['accuracy']:>10.4f}")
print(f"{'ECE':<25} {base_metrics['ece']:>10.4f} {sft_metrics['ece']:>10.4f} {dpo_metrics['ece']:>10.4f}")
print(f"{'Brier Score':<25} {base_metrics['brier']:>10.4f} {sft_metrics['brier']:>10.4f} {dpo_metrics['brier']:>10.4f}")
print(f"{'Overconfidence Rate':<25} {base_metrics['overconfidence_rate']:>10.4f} {sft_metrics['overconfidence_rate']:>10.4f} {dpo_metrics['overconfidence_rate']:>10.4f}")
