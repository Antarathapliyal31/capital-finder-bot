import torch
import math
import numpy as np
import json
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---- Define helper functions FIRST ----

def check_answer(model_answer, aliases):
    """Check if model's answer matches any ground truth alias"""
    # Clean model answer - take first line, first sentence
    model_answer = model_answer.lower().strip()
    model_answer = model_answer.split('\n')[0].strip()

    for alias in aliases:
        alias_clean = alias.lower().strip()
        # Check both directions - alias in answer OR answer in alias
        # This catches cases where model says "Pope Adrian IV"
        # and alias is "Adrian IV"
        if alias_clean in model_answer or model_answer.split('.')[0].strip() in alias_clean:
            return 1
    return 0
# ---- Setup device ----
# Use CPU with float32 to avoid MPS precision issues
# MPS + float16 can produce NaN/zero in softmax, which broke the previous run

device = torch.device("cpu")
dtype = torch.float32
print(f"Using device: {device}, dtype: {dtype}")

# ---- Load model ----

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype
)
# No .to(device) needed — CPU is default

print(f"Model loaded on {device}")

# ---- Quick single-question debug test ----

print("\n--- DEBUG TEST: Running one question to verify everything works ---")

test_prompt = "Question: What is the capital of France?\nAnswer:"
test_inputs = tokenizer(test_prompt, return_tensors="pt")

with torch.no_grad():
    test_outputs = model.generate(
        **test_inputs,
        max_new_tokens=20,
        output_scores=True,
        return_dict_in_generate=True,
        do_sample=False
    )

# Check generated text
test_gen_ids = test_outputs.sequences[0, test_inputs.input_ids.shape[1]:]
test_gen_text = tokenizer.decode(test_gen_ids, skip_special_tokens=True)
print(f"Generated text: {test_gen_text}")

# Check scores
print(f"Number of score tensors: {len(test_outputs.scores)}")
print(f"Score tensor shape: {test_outputs.scores[0].shape}")
print(f"Score tensor dtype: {test_outputs.scores[0].dtype}")

# Check first token probability
first_scores = test_outputs.scores[0]
first_probs = torch.softmax(first_scores.float(), dim=-1)
first_chosen = test_outputs.sequences[0, test_inputs.input_ids.shape[1]]
first_prob = first_probs[0, first_chosen].item()
print(f"First token: '{tokenizer.decode(first_chosen)}'")
print(f"First token probability: {first_prob}")
print(f"Sum of first probs (should be ~1.0): {first_probs.sum().item()}")
print(f"Max prob: {first_probs.max().item()}")
print(f"Any NaN in probs: {torch.isnan(first_probs).any().item()}")

# Compute full confidence for test
test_token_probs = []
for step, scores in enumerate(test_outputs.scores):
    probs = torch.softmax(scores.float(), dim=-1)
    chosen_token = test_outputs.sequences[0, test_inputs.input_ids.shape[1] + step]
    token_prob = probs[0, chosen_token].item()
    test_token_probs.append(token_prob)

print(f"All token probs: {test_token_probs}")
print(f"Any zero probs: {any(p == 0 for p in test_token_probs)}")

if len(test_token_probs) > 0 and all(p > 0 for p in test_token_probs):
    log_probs = [math.log(p) for p in test_token_probs]
    confidence = math.exp(sum(log_probs) / len(log_probs))
    print(f"Confidence: {confidence:.4f}")
    print("--- DEBUG TEST PASSED! Proceeding to full run ---\n")
else:
    print("--- DEBUG TEST FAILED! Token probs are zero or empty ---")
    print("This means the model/device setup has an issue.")
    exit(1)

# ---- Load TriviaQA ----

dataset = load_dataset("trivia_qa", "unfiltered", split="validation")
test_questions = dataset.select(range(2000))  # Start with 50, then scale to 2000
print(f"Loaded {len(test_questions)} questions")

# ---- Run baseline calibration ----

results = []

for i, example in enumerate(test_questions):
    question = example['question']
    ground_truth_aliases = example['answer']['aliases']

    # Format the prompt
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate with scores
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False
        )

    # Extract generated text
    generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute internal confidence
    token_probs = []
    for step, scores in enumerate(outputs.scores):
        probs = torch.softmax(scores.float(), dim=-1)
        chosen_token = outputs.sequences[0, inputs.input_ids.shape[1] + step]
        token_prob = probs[0, chosen_token].item()
        token_probs.append(token_prob)

    # Skip if no tokens generated
    if len(token_probs) == 0:
        print(f"  WARNING: Question {i} - no tokens generated, skipping")
        continue

    # Filter out zero probabilities
    valid_probs = [p for p in token_probs if p > 0]
    if len(valid_probs) == 0:
        print(f"  WARNING: Question {i} - all token probs are zero, skipping")
        continue

    # Geometric mean
    log_probs = [math.log(p) for p in valid_probs]
    confidence = math.exp(sum(log_probs) / len(log_probs))

    # Check correctness
    correct = check_answer(generated_text, ground_truth_aliases)

    results.append({
        'question': question,
        'generated': generated_text,
        'confidence': confidence,
        'correct': correct
    })

    # Print progress every 10 questions
    if i % 10 == 0:
        print(f"Processed {i}/{len(test_questions)} | "
              f"Answer: {generated_text[:50]}... | "
              f"Confidence: {confidence:.4f} | "
              f"Correct: {correct}")

# ---- Save results ----

with open('baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nDone! Processed {len(results)} questions successfully.")

if len(results) == 0:
    print("ERROR: No results collected. Something is still wrong.")
    exit(1)

print(f"Overall accuracy: {np.mean([r['correct'] for r in results]):.4f}")
print(f"Mean confidence: {np.mean([r['confidence'] for r in results]):.4f}")
print(f"Min confidence: {min(r['confidence'] for r in results):.4f}")
print(f"Max confidence: {max(r['confidence'] for r in results):.4f}")

# ---- Calibration Plot ----

confidences = [r['confidence'] for r in results]
corrects = [r['correct'] for r in results]

num_bins = 10
bins = np.linspace(0, 1, num_bins + 1)
bin_midpoints = []
bin_accuracies = []
bin_counts = []

for b in range(num_bins):
    mask = [(c >= bins[b] and c < bins[b + 1]) for c in confidences]
    bin_corrects = [c for c, m in zip(corrects, mask) if m]
    if len(bin_corrects) > 0:
        bin_midpoints.append((bins[b] + bins[b + 1]) / 2)
        bin_accuracies.append(np.mean(bin_corrects))
        bin_counts.append(len(bin_corrects))

# Print bin details so we can see the distribution
print("\n--- Calibration Bin Details ---")
for b in range(len(bin_midpoints)):
    print(f"  Bin {bin_midpoints[b]:.2f}: {bin_counts[b]} questions, "
          f"accuracy = {bin_accuracies[b]:.4f}")

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.plot(bin_midpoints, bin_accuracies, 'ro-', markersize=10, linewidth=2,
         label='Base model (Phi-2)')
# Add count labels on each point
for b in range(len(bin_midpoints)):
    plt.annotate(f'n={bin_counts[b]}',
                 (bin_midpoints[b], bin_accuracies[b]),
                 textcoords="offset points", xytext=(10, 10),
                 fontsize=9)
plt.xlabel('Internal Confidence (geometric mean token prob)')
plt.ylabel('Actual Accuracy')
plt.title(f'Baseline Calibration - {model_name}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('baseline_calibration.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved calibration plot to baseline_calibration.png")

# ---- ECE ----

ece = 0
total = len(confidences)
for b in range(len(bin_midpoints)):
    ece += (bin_counts[b] / total) * abs(bin_accuracies[b] - bin_midpoints[b])
print(f"\nECE: {ece:.4f}")

# ---- Overconfident errors ----

overconfident_errors = sum(1 for c, cor in zip(confidences, corrects)
                          if c >= 0.9 and cor == 0)
total_high_conf = sum(1 for c in confidences if c >= 0.9)
if total_high_conf > 0:
    print(f"Overconfident errors: {overconfident_errors}/{total_high_conf} "
          f"({overconfident_errors / total_high_conf * 100:.1f}%)")
else:
    print("No high-confidence (>=0.9) predictions found")

# ---- Confidence distribution ----

plt.figure(figsize=(8, 4))
plt.hist(confidences, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Internal Confidence')
plt.ylabel('Count')
plt.title(f'Confidence Distribution - {model_name}')
plt.axvline(x=np.mean(confidences), color='r', linestyle='--',
            label=f'Mean = {np.mean(confidences):.3f}')
plt.legend()
plt.savefig('confidence_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved confidence distribution to confidence_distribution.png")