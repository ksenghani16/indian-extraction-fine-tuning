# Indian Document Extractor

![Status](https://img.shields.io/badge/status-in%20progress-orange)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/YOUR-REPO/blob/main/notebooks/train_on_colab.ipynb)

Fine-tune a small open language model (Qwen2.5-1.5B-Instruct, LoRA) to turn messy Indian documents
(UPI SMS, GST invoices, addresses, job posts) into clean JSON, and measure honestly how it compares
with the untuned model.

    schema (the fields you want) + messy text   ->   model   ->   JSON

The schema is part of the prompt, so one model can handle document types it never saw in training.

> **Status: work in progress.** The v1 pipeline runs end to end. I found a real failure in v1
> (the model invents values for fields that are missing from the document), so v2 retrains with
> missing fields. v2 results are not in yet. See [Roadmap](#roadmap).

## Results so far (v1, before the missing-field fix)

200 documents per test set, base model vs LoRA-tuned model, trained on a Colab GPU.

| Test set | Metric | Base | Tuned |
|---|---|---|---|
| Held-out schemas (electricity bills, train tickets; never trained on) | Field F1 | 0.977 | 0.990 |
| | Exact match | 88.5% | 94.5% |
| | Hallucination rate | 1.0% | 0.0% |
| Same schemas as training (synthetic templates) | Exact match | 41.5% | 100% |
| | Null accuracy | 74.6% | 100% |

**How to read this:** the same-schema test uses the same templates as training, so 100% shows the
pipeline works, not that the model generalises. The held-out result is the more meaningful one, but
those two schemas are simple, so the base model was already strong. A hand-written test set and
harder held-out schemas are on the roadmap.

## What I found by testing it by hand

After training, I tried messages that did not match my templates:

- A UPI alert with **no date** came back with an **invented date** (`2021-06-15`).
- A message with **no account number** came back with the **balance copied into `account_last4`**.

**Cause:** in v1 training data, most fields were always present, so the model never learned that
`null` is a valid answer for them. My scorer also skipped date fields, so it would not have caught this.

**Fix (v2):** every field can now be missing (about 26% of training fields are null), a new
`test_missing` set measures it, more UPI wording variety is added, and the scorer now checks that a
predicted date's day, month and year actually appear in the document.

## Metrics

Valid-JSON rate, field-level precision/recall/F1, exact match, null accuracy (says `null` when the
value is missing), and hallucination rate (predicted values not supported by the document).
Results are reported per schema and per noise level (clean, light, heavy typos/OCR-style errors).

## Project structure

    schemas.py         document types and the exact prompt format
    generate_data.py   synthetic documents + correct JSON, random missing fields
    noise.py           typos, OCR-style errors, odd line breaks
    train_lora.py      LoRA fine-tuning
    evaluate.py        scores a model on a test file
    metrics.py         scoring maths
    modeling.py        model loading (Intel XPU / NVIDIA / CPU)
    extract.py         run the model on one document
    app.py             small Gradio demo
    notebooks/         Colab notebook for training

## Quick start

    pip install -r requirements.txt            # install PyTorch for your hardware first
    python generate_data.py --n_train 3000
    python evaluate.py --test data/test_heldout.jsonl --backend hf     # base model scores
    python train_lora.py --epochs 2            # use a GPU; see the Colab notebook
    python evaluate.py --test data/test_heldout.jsonl --backend hf --adapter outputs/adapter
    python extract.py --schema upi_sms --text "Rs.499.00 debited from A/c XX1234 on 03-Nov-26 to SWIGGY."

Detailed beginner instructions and troubleshooting: [SETUP.md](SETUP.md).
To train for free, open the notebook with the Colab badge above (runtime type: GPU).

## Roadmap

- [x] Synthetic data generator with noise (4 training + 2 held-out document types)
- [x] LoRA training and evaluation pipeline
- [x] v1 results, and failure analysis from manual testing
- [x] v2 data: missing fields, more wording variety, date-aware scoring
- [ ] Retrain on v2 data and report `test_missing` results
- [ ] Hand-written test set of real-style messy documents (never trained on)
- [ ] Harder held-out schemas (multiple line items, longer text, Hinglish)
- [ ] Compare against an API model (accuracy, cost, speed)
- [ ] Ablations: training-set size, LoRA rank
- [ ] Publish adapter and dataset on Hugging Face

## Limitations

- All data is **synthetic**. GSTINs and pincodes are well-formed but fake. No real personal documents are used.
- Dates are read **day first** (12/11/2026 means 12 November), following the generator's convention.
- Results so far come from template-based test sets, not real-world documents.
- Not for legal, tax or financial decisions without human review.

## Notes

- Training was done on a free Colab GPU. Integrated Intel Arc graphics could run models, but training
  on it failed, and for a small 0.5B model on 5 test documents the CPU was faster than the GPU (7.7 s vs 17.9 s per document). That is a very small sample, so treat it as a rough observation.
- Base model: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). Check its license before reusing the fine-tuned weights.

## License

To be added.
