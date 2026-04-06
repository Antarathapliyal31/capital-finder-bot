import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader

# ---- Config ----
device = torch.device("cpu")
dtype = torch.float32
model_name = "microsoft/phi-2"
max_seq_len = 256
batch_size = 4
grad_accum = 4
epochs = 3
lr = 2e-4
warmup_ratio = 0.03
weight_decay = 0.01

# ---- Load tokenizer and model ----
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

# ---- Apply LoRA ----
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---- Load SFT dataset ----
with open("sft_dataset.json", "r") as f:
    sft_data = json.load(f)

print(f"Loaded {len(sft_data)} SFT examples")


class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.examples = []
        for item in data:
            prompt = item["prompt"]
            target = item["target"]
            full_text = prompt + " " + target

            # Tokenize full sequence
            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = full_enc.input_ids.squeeze(0)
            attention_mask = full_enc.attention_mask.squeeze(0)

            # Tokenize prompt alone to find where target starts
            prompt_enc = tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_enc.input_ids.shape[1]

            # Labels: mask prompt tokens and padding with -100
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


dataset = SFTDataset(sft_data, tokenizer, max_seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---- Optimizer and scheduler ----
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=lr,
    weight_decay=weight_decay,
)
total_steps = (len(dataloader) // grad_accum) * epochs
warmup_steps = int(total_steps * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# ---- Training loop ----
model.train()
global_step = 0

for epoch in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / grad_accum
        loss.backward()

        epoch_loss += outputs.loss.item()
        num_batches += 1

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 20 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch+1}/{epochs} | Step {global_step} | Loss: {avg_loss:.4f}")

    # Handle remaining gradients
    if (step + 1) % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{epochs} complete | Avg Loss: {avg_loss:.4f}")

# ---- Save LoRA adapter ----
model.save_pretrained("./sft_lora_adapter")
tokenizer.save_pretrained("./sft_lora_adapter")
print("Saved SFT LoRA adapter to ./sft_lora_adapter/")

# ---- Sanity check ----
print("\n--- Sanity Check: 5 test questions ---")
model.eval()

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
        output = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
        )
    gen_ids = output[0, inputs.input_ids.shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"Q: {q}")
    print(f"A: {gen_text.strip()}\n")
