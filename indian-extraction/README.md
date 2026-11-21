# Indian Document Extractor (fine-tuning project)

Teach a small open language model to turn messy Indian documents
(UPI SMS, GST invoices, addresses, job posts) into clean JSON,
then **measure honestly** how it compares to the untuned model and to an API model.

Plain-English idea:

    schema (what to extract) + messy text   -->   model   -->   clean JSON

We put the *schema inside the prompt*, so the same model can handle
document types it never saw in training. Testing that is the unique part.

---------------------------------------------------------------------
## 1. File map (what each file does)

    indian-extraction/
    |-- requirements.txt   libraries to install
    |-- schemas.py         the document types + the exact prompt format
    |-- noise.py           makes text messy (typos, OCR errors, odd line breaks)
    |-- generate_data.py   STEP 1: creates fake documents + correct answers
    |-- train_lora.py      STEP 3: fine-tunes the model (LoRA)
    |-- evaluate.py        STEP 2 and 4: scores a model on the test files
    |-- metrics.py         the scoring maths (F1, hallucination, ...)
    |-- modeling.py        loads a model on Intel Arc / NVIDIA / CPU
    |-- extract.py         try the model on one document from the terminal
    |-- app.py             a small web demo (Gradio)
    |-- data/              (created by step 1) train / validation / test files
    |-- outputs/           (created by step 3) your trained adapter
    `-- results/           (created by evaluate.py) scores you can quote

## 2. One-time setup

1. Install Python 3.10+ and make a virtual environment:

       python -m venv venv
       venv\Scripts\activate          (Windows)
       source venv/bin/activate       (Linux/macOS)

2. Install PyTorch with Intel Arc (XPU) support. The exact command changes
   over time, so copy it from Intel's / PyTorch's official install page for XPU.

3. Check your GPU is visible:

       python -c "import torch; print(torch.xpu.is_available())"

   `True` = good. `False` = the project still runs on CPU (slowly), or run
   the training step on Google Colab / Kaggle.

4. Install the rest:

       pip install -r requirements.txt

## 3. Run it, step by step

### Step 1 - make the data (takes seconds)

    python generate_data.py --n_train 3000

Creates `data/`. Look at `data/train_full.jsonl` in a text editor to see what
examples look like. Files:

- `train_sft.jsonl`, `val_sft.jsonl`  used for training
- `test_in_schema.jsonl`  new documents from the SAME 4 document types
- `test_heldout.jsonl`    documents from 2 types the model NEVER trains on
                          (electricity bills, train tickets)

### Step 2 - check the scoring works, then get your "before" numbers

    python evaluate.py --test data/test_in_schema.jsonl --backend gold

Should print about 100% - it just proves the scoring code works. Then the
untuned model (your BEFORE score):

    python evaluate.py --test data/test_heldout.jsonl --backend hf --model Qwen/Qwen2.5-1.5B-Instruct
    python evaluate.py --test data/test_in_schema.jsonl --backend hf --model Qwen/Qwen2.5-1.5B-Instruct

### Step 3 - fine-tune

First a tiny smoke test, to catch errors in minutes instead of hours:

    python train_lora.py --max_samples 100 --epochs 1

If that finishes, run the real thing:

    python train_lora.py --epochs 2

Result: `outputs/adapter/` (small, because LoRA only saves the changes).

### Step 4 - your "after" numbers

    python evaluate.py --test data/test_heldout.jsonl  --backend hf --adapter outputs/adapter
    python evaluate.py --test data/test_in_schema.jsonl --backend hf --adapter outputs/adapter

Compare with step 2. The held-out result is the headline: it shows whether the
model learned extraction in general.

### Step 5 - compare with an API model (free tiers are fine)

Pick any provider with an OpenAI-compatible endpoint (Groq, Gemini, OpenAI...).
Set three environment variables (Windows: `set`, Linux/macOS: `export`):

    API_KEY=your_key
    API_BASE=https://provider.example/v1      (check your provider's docs)
    API_MODEL=model-name

    python evaluate.py --test data/test_heldout.jsonl --backend api --fewshot data/train_full.jsonl

It sends 3 example documents (from other schemas) with each request, so the
API model gets a fair few-shot prompt. Write down the model name and date.

### Step 6 - try it yourself

    python extract.py --schema upi_sms --text "Rs.499.00 debited from A/c XX1234 on 03-Nov-26 to SWIGGY. UPI Ref 409381772211."
    python app.py

## 4. How it works (the 5 ideas)

1. **Synthetic data.** We invent values, write them into templates, and so know
   the right JSON automatically. No labelling by hand.
2. **Noise.** Typos/OCR errors are added only to words that are NOT part of the
   answer, so the answer is always still in the text.
3. **Schema in the prompt.** The prompt has the field list + document. The model
   learns "read schema, copy/convert values from document, output JSON".
4. **LoRA.** The big model stays frozen; we train small add-on matrices.
   The loss is computed only on the JSON part of each example.
5. **Honest scoring.** `metrics.py` reports valid-JSON rate, field F1, exact match,
   null accuracy (does it say null instead of guessing?) and hallucination rate
   (values that are not in the document), split by schema and noise level.

## 5. IMPORTANT: make the evaluation credible

The synthetic test files come from the same templates as training, so scores on
`test_in_schema` will look good. For a resume-worthy result you also need:

- **A hand-written gold set (100-300 examples).** Write messy documents yourself
  (and ask friends), in the same JSON-lines format as `test_in_schema.jsonl`:
  `schema_name`, `noise` (0), `document`, `gold`, `prompt`, `completion`.
  Easiest: write a little script that takes your documents + gold dicts and
  builds `prompt` with `build_prompt(SCHEMAS[name], document)`.
  Never train on it. Then run `evaluate.py --test your_gold.jsonl`.
- **Held-out schemas** (already included) - the strongest evidence.
- Add your own held-out schema: put it in `SCHEMAS` + a generator, and keep it
  out of `TRAIN_SCHEMAS`.

## 6. Common problems

| Problem | Fix |
|---|---|
| `torch.xpu.is_available()` is False | Reinstall PyTorch using the official XPU instructions; update the Intel GPU driver; otherwise use Colab/Kaggle for training |
| Out of memory | `--batch 1 --grad_accum 16 --max_length 768`, or use a smaller model (`Qwen/Qwen2.5-0.5B-Instruct`) |
| `max_length` / argument errors in `SFTConfig` | TRL versions differ; rename `max_length` <-> `max_seq_length`, or `pip install -U trl` |
| Training is very slow | Normal on CPU. Use fewer examples first, or Colab/Kaggle |
| Model outputs extra text, not only JSON | The scorer extracts the first `{...}`; also check the model trained long enough |
| Download is slow / model asks for access | Some models need you to accept a license on huggingface.co and run `huggingface-cli login` |

## 7. Next steps (to make it stand out)

- Compare 500 vs 2000 vs 5000 training examples (`--n_train`) and LoRA rank 8/16/32 (`--rank`).
- Add constrained decoding (Outlines / llama.cpp grammar) and report JSON validity with vs without it.
- Measure speed on your Arc GPU (`avg_latency_sec` in the results) and compute cost per 1,000 docs vs the API's published list price.
- Export to GGUF or OpenVINO and publish the adapter + dataset on Hugging Face with a model card.
- Write up what failed. Error analysis with real examples impresses more than perfect numbers.

## 8. Notes on data and licenses

- All data is synthetic. GSTINs and pincodes are well-formed but fake. Never add real personal documents.
- Check the license of the base model before publishing your fine-tuned version.
