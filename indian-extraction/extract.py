"""
extract.py - use your trained model on ONE document.

  python extract.py --adapter outputs/adapter --schema upi_sms \
      --text "Rs.499.00 debited from A/c XX1234 on 03-Nov-26 to SWIGGY. UPI Ref 409381772211."
"""
import argparse, json
from schemas import SCHEMAS, build_prompt
from modeling import load_model, generate
from metrics import parse_json

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--adapter", default="outputs/adapter")
ap.add_argument("--schema", choices=list(SCHEMAS), required=True)
ap.add_argument("--text", required=True)
a = ap.parse_args()

tok, model, device = load_model(a.model, a.adapter)
raw = generate(tok, model, device, build_prompt(SCHEMAS[a.schema], a.text))
print(json.dumps(parse_json(raw), indent=2, ensure_ascii=False) if parse_json(raw) else "Invalid JSON:\n" + raw)
