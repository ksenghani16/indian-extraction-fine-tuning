"""
train_lora.py - fine-tunes a small model with LoRA.

LoRA = we freeze the big model and train tiny extra matrices ("adapters").
That is why it fits in a few GB of GPU memory.

Quick smoke test (2 minutes, checks everything works):
    python train_lora.py --max_samples 100 --epochs 1
Real run:
    python train_lora.py --epochs 2
"""
import argparse, torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling import get_device

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")  # check its license first
ap.add_argument("--train", default="data/train_sft.jsonl")
ap.add_argument("--val", default="data/val_sft.jsonl")
ap.add_argument("--out", default="outputs/adapter")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--grad_accum", type=int, default=8)
ap.add_argument("--max_length", type=int, default=1024)
ap.add_argument("--max_samples", type=int, default=None, help="use only N examples (for testing)")
a = ap.parse_args()

device = get_device()
print("Training on device:", device)

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float32)

data = load_dataset("json", data_files={"train": a.train, "val": a.val})
if a.max_samples:
    data["train"] = data["train"].select(range(min(a.max_samples, len(data["train"]))))
    data["val"] = data["val"].select(range(min(20, len(data["val"]))))

lora = LoraConfig(
    r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

cfg = SFTConfig(
    output_dir=a.out + "_checkpoints",
    num_train_epochs=a.epochs,
    per_device_train_batch_size=a.batch,
    gradient_accumulation_steps=a.grad_accum,
    learning_rate=a.lr,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    max_length=a.max_length,            # (older TRL versions call this max_seq_length)
    logging_steps=10,
    eval_strategy="steps", eval_steps=100,
    save_steps=100, save_total_limit=2,
    bf16=(device != "cpu"),
    use_cpu=(device == "cpu"),
    report_to="none",
    loss_type="nll",
    # data has "prompt" and "completion" columns -> loss is computed on the completion only
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=data["train"],
                     eval_dataset=data["val"], peft_config=lora, processing_class=tok)
trainer.train()
trainer.save_model(a.out)
tok.save_pretrained(a.out)
print("Saved LoRA adapter to", a.out)
