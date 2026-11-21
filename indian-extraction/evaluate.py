"""
evaluate.py - scores a model on a test file.

Examples:
  # 1. check the scoring pipeline works (uses the answers themselves, should give ~100%)
  python evaluate.py --test data/test_in_schema.jsonl --backend gold

  # 2. the untuned base model (your 'before' number)
  python evaluate.py --test data/test_heldout.jsonl --backend hf --model Qwen/Qwen2.5-1.5B-Instruct

  # 3. your fine-tuned model (your 'after' number)
  python evaluate.py --test data/test_heldout.jsonl --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/adapter

  # 4. an API model (OpenAI-compatible: Groq, Gemini, OpenAI ...). Set API_KEY, API_BASE, API_MODEL env vars
  python evaluate.py --test data/test_heldout.jsonl --backend api --fewshot data/train_full.jsonl
"""
import argparse, json, os, time, random
from collections import defaultdict
from metrics import parse_json, score_example, summarize
from schemas import SCHEMAS

ap = argparse.ArgumentParser()
ap.add_argument("--test", required=True)
ap.add_argument("--backend", choices=["gold", "hf", "api"], default="hf")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--adapter", default=None)
ap.add_argument("--fewshot", default=None, help="train_full.jsonl to take examples from (api backend)")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--max_examples", type=int, default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.test, encoding="utf-8")]
if a.max_examples:
    rows = rows[:a.max_examples]

# ---------------------------------------------------------------- backends
if a.backend == "hf":
    from modeling import load_model, generate
    tok, model, device = load_model(a.model, a.adapter)
    predict = lambda r: generate(tok, model, device, r["prompt"])
elif a.backend == "api":
    import requests
    key, base, name = os.environ["API_KEY"], os.environ["API_BASE"], os.environ["API_MODEL"]
    pool = [json.loads(l) for l in open(a.fewshot, encoding="utf-8")] if a.fewshot else []
    def predict(r):
        shots = [p for p in pool if p["schema_name"] != r["schema_name"]]
        random.Random(0).shuffle(shots)
        msgs = []
        for s in shots[:a.k]:
            msgs += [{"role": "user", "content": s["prompt"]}, {"role": "assistant", "content": s["completion"]}]
        msgs.append({"role": "user", "content": r["prompt"]})
        resp = requests.post(f"{base}/chat/completions", timeout=120,
                             headers={"Authorization": f"Bearer {key}"},
                             json={"model": name, "messages": msgs, "temperature": 0})
        return resp.json()["choices"][0]["message"]["content"]
else:
    predict = lambda r: r["completion"]

# ---------------------------------------------------------------- run
per_doc, valid, latencies, records = [], [], [], []
for i, r in enumerate(rows):
    t0 = time.time()
    raw = predict(r)
    latencies.append(time.time() - t0)
    pred = parse_json(raw)
    schema = SCHEMAS[r["schema_name"]]
    per_doc.append(score_example(pred, r["gold"], schema, r["document"]))
    valid.append(pred is not None)
    records.append({"schema": r["schema_name"], "noise": r.get("noise"), "raw": raw})
    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{len(rows)} done")

report = {"overall": summarize(per_doc, valid),
          "avg_latency_sec": sum(latencies) / len(latencies)}
for key in ("schema", "noise"):
    groups = defaultdict(list)
    for idx, rec in enumerate(records):
        groups[rec[key]].append(idx)
    report["by_" + key] = {str(g): summarize([per_doc[i] for i in ix], [valid[i] for i in ix])
                           for g, ix in sorted(groups.items(), key=lambda kv: str(kv[0]))}

def show(title, m):
    na = "n/a" if m["null_accuracy"] is None else f"{m['null_accuracy']:.3f}"
    print(f"{title:<22} n={m['n']:<4} valid={m['json_valid']:.3f} F1={m['f1']:.3f} "
          f"exact={m['exact_match']:.3f} null_acc={na} halluc={m['hallucination']:.3f}")

print("\n=== RESULTS ===")
show("OVERALL", report["overall"])
for k, m in report["by_schema"].items(): show("schema: " + k, m)
for k, m in report["by_noise"].items(): show("noise level: " + k, m)
print(f"avg latency per doc: {report['avg_latency_sec']:.2f}s")

out = a.out or f"results/{a.backend}_{os.path.basename(a.test).replace('.jsonl', '')}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"report": report, "predictions": records}, open(out, "w"), indent=2, ensure_ascii=False)
print("Saved", out)
