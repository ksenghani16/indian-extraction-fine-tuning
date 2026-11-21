"""
modeling.py - small helpers to load a model and generate text.
Picks the best device: Intel Arc (xpu) -> NVIDIA (cuda) -> CPU.
Set the environment variable DEVICE=cpu (or xpu / cuda) to force one.
"""
import os
import torch


def get_device() -> str:
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str, adapter: str = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = get_device()
    if device == "xpu" or (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8):
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16             # older NVIDIA GPUs (T4, P100)
    else:
        dtype = torch.float32
    print("Device:", device, "| dtype:", dtype)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()      # fuse LoRA into the model = faster inference
    model.to(device).eval()
    return tok, model, device


@torch.no_grad()
def generate(tok, model, device, prompt: str, max_new_tokens: int = 200) -> str:
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)