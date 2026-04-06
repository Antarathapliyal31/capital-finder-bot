import torch
import torch.nn.functional as F
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader

# ---- Config ----
device = torch.device("cpu")
dtype = torch.float32
model_name = "microsoft/phi-2"
max_seq_len = 256
batch_size = 4
grad_accum = 4
epochs = 3
lr = 5e-5
warmup_ratio = 0.03
beta = 0.1

# ---- Load tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ---- Load policy model (trainable) ----
print("Loading policy model...")
policy_base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
policy_model = PeftModel.from_pretrained(policy_base, "./sft_lora_adapter")
# Unfreeze LoRA weights
for name, param in policy_model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True

# ---- Load reference model (frozen) ----
print("Loading reference model...")
ref_base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
ref_model = PeftModel.from_pretrained(ref_base, "./sft_lora_adapter")
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False
print("Both models loaded.")

# ---- Load DPO pairs ----
with open("dpo_pairs.json", "r") as f:
    dpo_data = json.load(f)

print(f"Loaded {len(dpo_data)} DPO pairs")


class DPODataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.examples = []
        for item in data:
            prompt = item["prompt"]
            winner = item["winner"]
            loser = item["loser"]

            # Tokenize prompt to get prompt length
            prompt_enc = tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_enc.input_ids.shape[1]

            # Tokenize prompt + winner
            winner_full = prompt + " " + winner
            winner_enc = tokenizer(
                winner_full, truncation=True, max_length=max_len,
                padding="max_length", return_tensors="pt",
            )

            # Tokenize prompt + loser
            loser_full = prompt + " " + loser
            loser_enc = tokenizer(
                loser_full, truncation=True, max_length=max_len,
                padding="max_length", return_tensors="pt",
            )

            self.examples.append({
                "winner_input_ids": winner_enc.input_ids.squeeze(0),
                "winner_attention_mask": winner_enc.attention_mask.squeeze(0),
                "loser_input_ids": loser_enc.input_ids.squeeze(0),
                "loser_attention_mask": loser_enc.attention_mask.squeeze(0),
                "prompt_len": prompt_len,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


dataset = DPODataset(dpo_data, tokenizer, max_seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def compute_target_logprobs(model, input_ids, attention_mask, prompt_len):
    """Compute sum of log probabilities for target tokens only."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch, seq_len, vocab)

    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].clone()

    # Mask prompt tokens (positions 0..prompt_len-2 in shifted space)
    # In shifted labels, position i corresponds to predicting token i+1
    # Prompt tokens are 0..prompt_len-1, so in shifted space mask 0..prompt_len-2
    for b in range(shift_mask.shape[0]):
        shift_mask[b, :prompt_len - 1] = 0

    # Log softmax
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probs of actual tokens
    gathered = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Sum over target tokens only
    target_logprobs = (gathered * shift_mask.float()).sum(dim=-1)
    return target_logprobs


# ---- Optimizer and scheduler ----
trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
total_steps = (len(dataloader) // grad_accum) * epochs
warmup_steps = int(total_steps * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# ---- Training loop ----
policy_model.train()
global_step = 0

for epoch in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        w_ids = batch["winner_input_ids"].to(device)
        w_mask = batch["winner_attention_mask"].to(device)
        l_ids = batch["loser_input_ids"].to(device)
        l_mask = batch["loser_attention_mask"].to(device)
        prompt_len = batch["prompt_len"][0].item()  # same for all in batch

        # Reference model logprobs (no grad)
        with torch.no_grad():
            ref_w_logprobs = compute_target_logprobs(ref_model, w_ids, w_mask, prompt_len)
            ref_l_logprobs = compute_target_logprobs(ref_model, l_ids, l_mask, prompt_len)

        # Policy model logprobs
        policy_w_logprobs = compute_target_logprobs(policy_model, w_ids, w_mask, prompt_len)
        policy_l_logprobs = compute_target_logprobs(policy_model, l_ids, l_mask, prompt_len)

        # DPO loss
        margin = beta * (
            (policy_w_logprobs - ref_w_logprobs)
            - (policy_l_logprobs - ref_l_logprobs)
        )
        loss = -F.logsigmoid(margin).mean() / grad_accum
        loss.backward()

        epoch_loss += loss.item() * grad_accum
        num_batches += 1

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 10 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch+1}/{epochs} | Step {global_step} | Loss: {avg_loss:.4f}")

    # Handle remaining gradients
    if (step + 1) % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{epochs} complete | Avg Loss: {avg_loss:.4f}")

# ---- Save DPO adapter ----
policy_model.save_pretrained("./dpo_lora_adapter")
tokenizer.save_pretrained("./dpo_lora_adapter")
print("Saved DPO LoRA adapter to ./dpo_lora_adapter/")

# ---- Sanity check ----
print("\n--- Sanity Check: 5 test questions ---")
policy_model.eval()

test_questions = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the largest planet in our solar system?",
    "Who painted the Mona Lisa?",
    "What year did World War II end?",
]

for q in test_questions:
    prompt = f"Question: {q},\n Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = policy_model.generate(
            **inputs, max_new_tokens=40, do_sample=False,
        )
    gen_ids = output[0, inputs.input_ids.shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"Q: {q}")
    print(f"A: {gen_text.strip()}\n")
