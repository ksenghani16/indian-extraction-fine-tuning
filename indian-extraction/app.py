"""
app.py - a tiny web demo.   python app.py   then open the link it prints.
You can pick a built-in schema OR paste your own schema JSON (tests generalisation!).
"""
import json, gradio as gr
from schemas import SCHEMAS, build_prompt
from modeling import load_model, generate
from metrics import parse_json

MODEL, ADAPTER = "Qwen/Qwen2.5-1.5B-Instruct", "outputs/adapter"
tok, model, device = load_model(MODEL, ADAPTER)

def run(schema_name, custom_schema, text):
    try:
        schema = json.loads(custom_schema) if custom_schema.strip() else SCHEMAS[schema_name]
    except json.JSONDecodeError:
        return "Your custom schema is not valid JSON."
    raw = generate(tok, model, device, build_prompt(schema, text))
    obj = parse_json(raw)
    return json.dumps(obj, indent=2, ensure_ascii=False) if obj else "Model output was not valid JSON:\n" + raw

demo = gr.Interface(
    fn=run,
    inputs=[gr.Dropdown(list(SCHEMAS), value="upi_sms", label="Schema"),
            gr.Textbox(label="OR paste a custom schema (JSON: field -> type description)", lines=3),
            gr.Textbox(label="Document text", lines=8)],
    outputs=gr.Textbox(label="Extracted JSON", lines=12),
    title="Indian Document Extractor")
demo.launch()
