"""
metrics.py - turns (model output, correct answer) into numbers.

Reported:
  json_valid      share of outputs that are parseable JSON
  precision/recall/f1   field-level, over non-null gold fields
  exact_match     share of documents where EVERY field is right
  null_accuracy   when the true value is null, did the model also say null?
  hallucination   share of predicted values that do NOT appear in the document
"""
import json, re


def parse_json(text: str):
    """Find the first {...} in the model output. Returns dict or None."""
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _norm_str(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def equal(pred, gold) -> bool:
    if pred is None or gold is None:
        return pred is None and gold is None
    if isinstance(gold, list):
        return isinstance(pred, list) and {_norm_str(x) for x in pred} == {_norm_str(x) for x in gold}
    if isinstance(gold, (int, float)) and not isinstance(gold, bool):
        try:
            return abs(float(pred) - float(gold)) < 0.01
        except (TypeError, ValueError):
            return False
    return _norm_str(pred) == _norm_str(gold)


def _in_doc(value, doc: str, doc_nocomma: str) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        cands = {f"{value:.2f}", f"{value:g}", str(int(value))}
        return any(c in doc_nocomma for c in cands)
    return _norm_str(value) in _norm_str(doc)


def score_example(pred, gold: dict, schema: dict, doc: str):
    """Returns counts for one document."""
    c = dict(tp=0, fp=0, fn=0, null_total=0, null_ok=0, pred_checked=0, pred_halluc=0, exact=0)
    doc_nocomma = doc.replace(",", "")
    if pred is None:                         # invalid JSON: everything non-null is missed
        c["fn"] = sum(v is not None for v in gold.values())
        c["null_total"] = sum(v is None for v in gold.values())
        return c
    all_ok = True
    for field, g in gold.items():
        p = pred.get(field)
        if g is None:
            c["null_total"] += 1
            if p is None: c["null_ok"] += 1
            else: c["fp"] += 1; all_ok = False
        elif equal(p, g):
            c["tp"] += 1
        else:
            all_ok = False; c["fn"] += 1
            if p is not None: c["fp"] += 1
    # hallucination check (skip dates and enums: they are re-formatted, not copied)
    for field, p in pred.items():
        desc = schema.get(field, "")
        if p is None or "date" in desc or desc.startswith("one of"):
            continue
        items = p if isinstance(p, list) else [p]
        for it in items:
            c["pred_checked"] += 1
            if not _in_doc(it, doc, doc_nocomma):
                c["pred_halluc"] += 1
    c["exact"] = int(all_ok)
    return c


def summarize(per_doc, valid_flags):
    n = len(per_doc)
    tot = {k: sum(d[k] for d in per_doc) for k in per_doc[0]} if per_doc else {}
    tp, fp, fn = tot.get("tp", 0), tot.get("fp", 0), tot.get("fn", 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n": n,
        "json_valid": sum(valid_flags) / n if n else 0,
        "precision": prec, "recall": rec, "f1": f1,
        "exact_match": tot.get("exact", 0) / n if n else 0,
        "null_accuracy": tot["null_ok"] / tot["null_total"] if tot.get("null_total") else None,
        "hallucination": tot["pred_halluc"] / tot["pred_checked"] if tot.get("pred_checked") else 0,
    }
